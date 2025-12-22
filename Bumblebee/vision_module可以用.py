import cv2
import numpy as np

class VisionModule:
    def __init__(self, physics, width=320, height=240):
        self.width = width
        self.height = height
        
        self.cam_left_id = -1
        self.cam_right_id = -1
        self._find_camera_ids(physics)
        
        # 颜色阈值 (蓝色)
        self.color_config = {
            'blue': (np.array([100, 60, 20]), np.array([140, 255, 255])),
            'green': (np.array([35, 60, 20]), np.array([90, 255, 255])),
        }
        
        # === ⚙️ 控制参数 ===
        self.STOP_DIST = 0       # 停止距离 (m)
        self.SLOW_DOWN_DIST = 0.5   # 减速距离 (m)
        
        # 增益
        self.GAIN_VX = 0.4    # 前进
        self.GAIN_VY = 0.3    # 侧移 (降低增益防抖动)
        self.GAIN_WZ = 1.0    # 转向
        
        self.ALIGN_DEADZONE = 0.05 # 对准死区
        
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

            # === A. 计数 (只听左眼的，保持不变) ===
            if targets_l:
                targets_l.sort(key=lambda x: x['area'], reverse=True)
                output["colors"] = [t['color'] for t in targets_l]

            # === B. 导航 (关键修改：智能去重融合) ===
            # 以前是直接相加，现在要过滤
            merged_targets = self._merge_stereo_targets(targets_l, targets_r)
            
            if merged_targets:
                output["is_found"] = True 
                output["raw_targets"] = merged_targets

                # 选最近的
                best_target = min(merged_targets, key=lambda x: x['dist'])
                
                dist = best_target['dist']
                bias_x = best_target['bias_x']
                
                # --- 速度控制 (保持之前的逻辑) ---
                # 1. Vx
                dist_error = dist - self.STOP_DIST
                if dist_error <= 0:
                    output["vx"] = 0.0
                else:
                    speed_factor = dist_error / self.SLOW_DOWN_DIST
                    speed_factor = np.clip(speed_factor, 0.0, 1.0)
                    if dist < 0.5 and abs(bias_x) > 0.2:
                        speed_factor *= 0.5
                    output["vx"] = speed_factor * self.GAIN_VX
                    if output["vx"] > 0 and output["vx"] < 0.01:
                        output["vx"] = 0.01

                # 2. Omega
                if abs(bias_x) > self.ALIGN_DEADZONE:
                    output["omega"] = -bias_x * self.GAIN_WZ
                else:
                    output["omega"] = 0.0

                # 3. Vy
                if dist < 1.0 and abs(bias_x) > self.ALIGN_DEADZONE:
                    output["vy"] = -bias_x * self.GAIN_VY
                else:
                    output["vy"] = 0.0

            # Debug 显示
            vis_img = np.hstack((img_l, img_r))
            self._draw_debug_overlay(vis_img, output)
            if 'best_target' in locals():
                tx, ty, tw, th = best_target['rect']
                cx, cy = tx + tw // 2, ty + th // 2
                offset = 0 if best_target['eye'] == 'L' else self.width
                cv2.circle(vis_img, (cx + offset, cy), 5, (0, 0, 255), 3)

            cv2.imshow('Vision_Debug', vis_img)
            cv2.waitKey(1)
            
            return output

    def _merge_stereo_targets(self, left_targets, right_targets):
        """
        双目去重融合：
        """
        final_list = list(left_targets) # 先把左眼看到的全部加入
        
        for r_obj in right_targets:
            is_duplicate = False
            for l_obj in left_targets:
                # 判断是否是同一个物体：
                # 1. 距离差距很小 (< 0.5m)
                # 2. 颜色相同
                dist_diff = abs(r_obj['dist'] - l_obj['dist'])
                if dist_diff < 0.5 and r_obj['color'] == l_obj['color']:
                    # 这是一个重复的目标（左眼已经看到了）
                    # 因为我们优先信赖左眼，所以直接忽略这个右眼数据
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # 只有当左眼完全没看到这个物体时（比如在极右侧），才采纳右眼数据
                final_list.append(r_obj)
                
        return final_list

    def _scan_colors(self, img, depth, is_left):
        """返回所有符合条件的目标列表"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections = []
        
        for color_name, (lower, upper) in self.color_config.items():
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学闭运算：非常重要！把"碎片"粘合成"整体"
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                area = cv2.contourArea(c)
                # 过滤小噪点
                if area > 20:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w // 2, y + h // 2
                    d = float(depth[cy, cx])
                    rel_x = (cx - (self.width/2)) / (self.width/2)
                    bias_x = -0.5 + rel_x * 0.5 if is_left else 0.5 + rel_x * 0.5
                    
                    det = {
                        'color': color_name,
                        'area': area,
                        'dist': d,
                        'bias_x': bias_x,
                        'rect': (x, y, w, h),
                        'eye': 'L' if is_left else 'R'
                    }
                    detections.append(det)
                    
                    # 画绿色框
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return detections

    def _draw_debug_overlay(self, img, data):
        status = "FOUND" if data['is_found'] else "SEARCHING"
        cv2.putText(img, f"[{status}] Alt:{data['altitude']:.2f}m", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cmd_text = f"Vx:{data['vx']:.2f} Vy:{data['vy']:.2f} Wz:{data['omega']:.2f}"
        cv2.putText(img, cmd_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示计数列表
        if data['colors']:
            # 只显示前3个，防止太长
            colors_str = f"Count(LeftEye): {len(data['colors'])} {data['colors'][:3]}"
            cv2.putText(img, colors_str, (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _render_eye(self, physics, cam_id):
        rgb = physics.render(width=self.width, height=self.height, camera_id=cam_id)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
        try: depth = physics.render(width=self.width, height=self.height, camera_id=cam_id, depth=True)
        except: depth = np.zeros((self.height, self.width))
        return rgb, depth