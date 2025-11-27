import os
os.environ["MUJOCO_GL"] = "osmesa"   # 或 "egl"，哪个能跑用哪个

import mujoco
import numpy as np
from PIL import Image

here = os.path.dirname(__file__)
xml_path = os.path.join(here, "fruitfly.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Model loaded, nq =", model.nq, "nv =", model.nv)

# 分辨率先用小一点，省资源
renderer = mujoco.Renderer(model, width=400, height=300)

# 先让系统跑一下
for _ in range(500):
    mujoco.mj_step(model, data)

# 你 XML 里存在的相机名字
cam_names = [
    "track1", "track2", "track3",
    "back", "side", "bottom", "hero",
    "eye_left", "eye_right",
]

for cam in cam_names:
    try:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        out_path = os.path.join(here, f"bee_view_{cam}.png")
        img.save(out_path)
        print("saved:", out_path)
    except mujoco.FatalError as e:
        print("camera failed:", cam, e)

renderer.close()
