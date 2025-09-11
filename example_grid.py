import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()

frames_combined = []
cameras = ["top", "left_wrist", "right_wrist", "front_close"]

for _ in range(100):  # 짧게 테스트
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 4개 카메라에서 동시 렌더링
    images = []
    for camera_name in cameras:
        image = env.unwrapped._env.physics.render(height=240, width=320, camera_id=camera_name)  # 크기 줄임
        images.append(image)
    
    # 2x2 그리드로 배치
    top_row = np.hstack([images[0], images[1]])      # top + left_wrist
    bottom_row = np.hstack([images[2], images[3]])   # right_wrist + front_close
    combined_image = np.vstack([top_row, bottom_row])
    
    frames_combined.append(combined_image)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# 2x2 그리드 영상 저장
print("Saving combined video...")
imageio.mimsave("combined_4cameras.mp4", np.stack(frames_combined), fps=25)
print("Done! Created combined_4cameras.mp4")