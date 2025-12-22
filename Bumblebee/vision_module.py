import cv2
import numpy as np

class VisionModule:
    def __init__(self, physics, width=320, height=240):
        self.width = width
        self.height = height
        
        self.cam_left_id = -1
        self.cam_right_id = -1
        self._find_camera_ids(physics)
        
        self.color_config = {
            'blue': (np.array([100, 60, 20]), np.array([140, 255, 255])),
            'green': (np.array([35, 60, 20]), np.array([90, 255, 255])),
        }
        
        # === ⚙️ 控制参数 (单向冲刺版) ===
        
        # 1. 停止距离
        # 如果你想"贴上"，可以把这个值设得非常小，比如 0.05
        # 配合 CHIN_OFFSET 使用
        self.STOP_DIST = 0.20
        
        # 2. 下巴补偿
        self.CHIN_OFFSET_X = 0.02  
        
        # 3. 速度控制
        self.SLOW_DOWN_DIST = 0.6   # 0.6米处开始减速，给够刹车距离
        self.GAIN_VX = 0.5          # 最大巡航速度
        
        # ⚠️ 这里的最小速度仅用于远距离！
        # 一旦进入最后 10cm，这个限制会被取消，允许完全停下
        self.MIN_CRUISE_SPEED = 0.05 
        
        self.GAIN_VY = 0.3    
        self.GAIN_WZ = 1.0    
        self.ALIGN_DEADZONE = 0.03 
        
        cv2.namedWindow('Vision_Debug', cv2.WINDOW_AUTOSIZE)

    def _find_camera_ids(self, physics):
        for i in range(physics.model.ncam):
            name = physics.model.id2name(i, 'camera')
            if not name: continue
            if 'left' in name and 'eye' in name: self.cam_left_id = i
            elif 'right' in name and 'eye' in name: self.cam_right_id = i

    def get_control_signals(self, physics):
        # 1. 渲染
        img_l, depth_l = self._render_eye(physics, self.cam_left_id)
        img_r, depth_r = self._render_eye(physics, self.cam_right_id)
        
        # 2. 识别
        targets_l = self._scan_colors(img_l, depth_l, is_left=True)
        targets_r = self._scan_colors(img_r, depth_r, is_left=False)
        
        output = {
            "vx": 0.0, "vy": 0.0, "altitude": 0.0, "omega": 0.0,
            "is_found": False, "colors": [], "raw_targets": []
        }
        output["altitude"] = physics.data.qpos[2]

        # A. 计数
        if targets_l:
            targets_l.sort(key=lambda x: x['area'], reverse=True)
            output["colors"] = [t['color'] for t in targets_l]

        # B. 导航
        merged_targets = self._merge_stereo_targets(targets_l, targets_r)
        
        if merged_targets:
            output["is_found"] = True 
            output["raw_targets"] = merged_targets

            best_target = min(merged_targets, key=lambda x: x['dist'])
            dist = best_target['dist']
            bias_x = best_target['bias_x']
            
            # ==========================================
            # 🚀 只有油门和刹车，没有倒档
            # ==========================================
            
            target_stop_dist = self.STOP_DIST - self.CHIN_OFFSET_X
            dist_error = dist - target_stop_dist
            
            # 1. 距离控制逻辑
            if dist_error <= 0.01:
                # 🛑 情况A: 到达或者飞过头了
                # 绝对不倒车！直接切断动力，依靠惯性停下或撞上
                output["vx"] = 0.0
            else:
                # 🚀 情况B: 还没到，计算向前速度
                
                # 线性减速曲线: error 越大速度越快
                speed_factor = dist_error / self.SLOW_DOWN_DIST 
                speed_factor = np.clip(speed_factor, 0.0, 1.0)
                
                # 基础速度
                output["vx"] = speed_factor * self.GAIN_VX
                
                # 🔎 防磨蹭逻辑 (智能版)
                # 只有当距离还比较远 (>0.1m) 时，才强制最小速度，防止半路停下
                # 一旦进入最后 0.1m，取消最小速度限制，允许它平滑减速到 0
                if dist_error > 0.1:
                    if output["vx"] < self.MIN_CRUISE_SPEED:
                        output["vx"] = self.MIN_CRUISE_SPEED

            # ⚠️ 再次强制约束：确保 vx 绝不为负
            output["vx"] = max(0.0, output["vx"])

            # 2. 旋转控制
            if abs(bias_x) > self.ALIGN_DEADZONE:
                output["omega"] = -bias_x * self.GAIN_WZ
            else:
                output["omega"] = 0.0

            # 3. 侧移控制
            if abs(bias_x) > self.ALIGN_DEADZONE:
                output["vy"] = -bias_x * self.GAIN_VY
            else:
                output["vy"] = 0.0

            # 4. 安全限速 (仅在未对准时减速)
            # 如果头歪得厉害，把速度降下来，给旋转留时间
            if dist < 0.5 and abs(bias_x) > 0.2:
                output["vx"] *= 0.5

        # Debug
        vis_img = np.hstack((img_l, img_r))
        self._draw_debug_overlay(vis_img, output)
        if 'best_target' in locals():
            tx, ty, tw, th = best_target['rect'] 
            cx, cy = int(best_target['cx']), int(best_target['cy'])
            offset = 0 if best_target['eye'] == 'L' else self.width
            cv2.drawMarker(vis_img, (cx + offset, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            cv2.circle(vis_img, (cx + offset, cy), 5, (0, 255, 255), 1)

        cv2.imshow('Vision_Debug', vis_img)
        cv2.waitKey(1)
        
        return output

    # ... (其余函数 _merge_stereo_targets, _scan_colors, _draw_debug_overlay, _render_eye 保持不变) ...
    # 为了节省篇幅，这里省略了未修改的辅助函数，请保留你之前文件里的这些函数！
    
    def _merge_stereo_targets(self, left_targets, right_targets):
        final_list = list(left_targets)
        for r_obj in right_targets:
            is_duplicate = False
            for l_obj in left_targets:
                dist_diff = abs(r_obj['dist'] - l_obj['dist'])
                if dist_diff < 0.5 and r_obj['color'] == l_obj['color']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_list.append(r_obj)
        return final_list

    def _scan_colors(self, img, depth, is_left):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections = []
        for color_name, (lower, upper) in self.color_config.items():
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 20:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        x, y, w, h = cv2.boundingRect(c)
                        cx, cy = x + w // 2, y + h // 2
                    cx = np.clip(cx, 0, self.width - 1)
                    cy = np.clip(cy, 0, self.height - 1)
                    x, y, w, h = cv2.boundingRect(c)
                    d = float(depth[cy, cx])
                    rel_x = (cx - (self.width/2)) / (self.width/2)
                    bias_x = -0.5 + rel_x * 0.5 if is_left else 0.5 + rel_x * 0.5
                    det = {
                        'color': color_name, 'area': area, 'dist': d, 'bias_x': bias_x,
                        'rect': (x, y, w, h), 'cx': cx, 'cy': cy, 'eye': 'L' if is_left else 'R'
                    }
                    detections.append(det)
                    box_color = (0, 255, 0) if color_name == 'green' else (255, 0, 0)
                    cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
        return detections

    def _draw_debug_overlay(self, img, data):
        status = "FOUND" if data['is_found'] else "SEARCHING"
        cv2.putText(img, f"[{status}] Alt:{data['altitude']:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cmd_text = f"Vx:{data['vx']:.2f} Vy:{data['vy']:.2f} Wz:{data['omega']:.2f}"
        cv2.putText(img, cmd_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if data['colors']:
            colors_str = f"Count: {len(data['colors'])} {data['colors'][:3]}"
            cv2.putText(img, colors_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _render_eye(self, physics, cam_id):
        rgb = physics.render(width=self.width, height=self.height, camera_id=cam_id)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
        try: depth = physics.render(width=self.width, height=self.height, camera_id=cam_id, depth=True)
        except: depth = np.zeros((self.height, self.width))
        return rgb, depth