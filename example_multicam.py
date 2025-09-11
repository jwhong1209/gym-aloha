import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()

# 4개 카메라별 프레임 저장
frames_top = []
frames_left_wrist = []
frames_right_wrist = []
frames_front_close = []

cameras = ["top", "left_wrist", "right_wrist", "front_close"]

for _ in range(100):  # 짧게 테스트
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 각 카메라에서 렌더링
    for camera_name in cameras:
        # TimeLimit 래퍼를 통해 접근
        image = env.unwrapped._env.physics.render(height=480, width=640, camera_id=camera_name)
        
        if camera_name == "top":
            frames_top.append(image)
        elif camera_name == "left_wrist":
            frames_left_wrist.append(image)
        elif camera_name == "right_wrist":
            frames_right_wrist.append(image)
        elif camera_name == "front_close":
            frames_front_close.append(image)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# 각 카메라별로 동영상 저장
print("Saving videos...")
imageio.mimsave("top_camera.mp4", np.stack(frames_top), fps=25)
imageio.mimsave("left_wrist_camera.mp4", np.stack(frames_left_wrist), fps=25)
imageio.mimsave("right_wrist_camera.mp4", np.stack(frames_right_wrist), fps=25)
imageio.mimsave("front_close_camera.mp4", np.stack(frames_front_close), fps=25)
print("Done! Created 4 videos.")