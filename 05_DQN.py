import numpy as np
from collections import deque


# ===================== DQN 标准预处理函数（纯数据逻辑） =====================
def preprocess_frame(frame):
    """
    DQN 帧预处理：模拟灰度化 + 像素归一化 [0,1]
    无OpenCV、无深度学习框架依赖
    """
    frame = frame / 255.0
    return frame


# ===================== 模拟数据采集与清洗 =====================
print("========== DQN 深度Q网络 - 数据采集 & 清洗 ==========")

# DQN 标准参数
single_frame_shape = (84, 84)  # 单帧尺寸
frame_stack = 4  # 连续4帧堆叠（DQN核心预处理）
max_episode = 2  # 模拟回合数
step_per_episode = 5  # 每回合步数

# 帧队列（实现帧堆叠）
frame_queue = deque(maxlen=frame_stack)
# 存储最终清洗后的完整状态数据
clean_state_list = []

for episode in range(max_episode):
    print(f"\n--- 开始第 {episode + 1} 回合数据采集 ---")
    # 1. 生成原始模拟图像帧
    raw_frame = np.random.randint(0, 255, size=single_frame_shape, dtype=np.float32)

    # 2. 预处理 + 填充初始队列
    for _ in range(frame_stack):
        processed = preprocess_frame(raw_frame)
        frame_queue.append(processed)

    # 拼接4帧为一个完整状态
    combine_state = np.concatenate(list(frame_queue)).flatten()
    clean_state_list.append(combine_state)
    print(f"初始状态维度: {combine_state.shape}")

    # 逐步采集数据
    for step in range(step_per_episode):
        # 生成新原始帧
        new_raw = np.random.randint(0, 255, size=single_frame_shape, dtype=np.float32)
        new_process = preprocess_frame(new_raw)
        frame_queue.append(new_process)

        # 拼接新状态
        new_state = np.concatenate(list(frame_queue)).flatten()
        clean_state_list.append(new_state)

# 数据清洗：过滤异常全0帧
valid_data = []
for state in clean_state_list:
    # 剔除全黑无效帧
    if np.sum(state) > 1e-3:
        valid_data.append(state)

# 输出统计结果
print("\n========== 数据清洗结果统计 ==========")
print(f"原始采集样本总数: {len(clean_state_list)}")
print(f"清洗后有效样本总数: {len(valid_data)}")
print("预处理流程：帧归一化 + 4帧堆叠 + 无效帧过滤 全部完成")
print("DQN 数据采集与清洗任务结束")