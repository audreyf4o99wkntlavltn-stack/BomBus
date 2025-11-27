import os
import mujoco

here = os.path.dirname(__file__)
xml_path = os.path.join(here, "fruitfly.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Model loaded OK, nq =", model.nq, "nv =", model.nv)

for i in range(1000):
    mujoco.mj_step(model, data)

print("Simulation stepped 1000 steps without error.")
