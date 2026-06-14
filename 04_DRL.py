import numpy as np


# ===================== 1. DRL 数据预处理/清洗函数 =====================
def normalize_state(state):
    """状态标准化（DRL通用清洗）"""
    mean = np.mean(state)
    std = np.std(state) + 1e-8
    return (state - mean) / std


def filter_abnormal_trajectory(reward_list):
    """过滤异常轨迹：极值奖励、无效回合"""
    valid_rewards = []
    for r in reward_list:
        # 过滤极端异常奖励
        if -200 < r < 200:
            valid_rewards.append(r)
    return valid_rewards


# ===================== 2. 模拟生成DRL离线轨迹数据 =====================
print("【深度强化学习 DRL】离线模拟数据采集与清洗")
# 模拟参数：状态维度、回合数
state_dim = 4
episodes = 8

# 存储轨迹：(状态, 动作, 奖励, 下一状态)
trajectory_data = []
all_rewards = []

for ep in range(episodes):
    # 随机生成初始状态
    state = np.random.randn(state_dim)
    state = normalize_state(state)

    total_reward = 0
    step_count = 10  # 每个回合固定10步

    for _ in range(step_count):
        action = np.random.randint(0, 2)  # 模拟动作 0/1
        reward = np.random.randint(-10, 20)  # 模拟单步奖励
        next_state = np.random.randn(state_dim)
        next_state = normalize_state(next_state)

        # 存入轨迹
        trajectory_data.append([state.tolist(), action, reward, next_state.tolist()])
        total_reward += reward
        state = next_state

    all_rewards.append(total_reward)
    print(f"回合 {ep + 1} 总奖励: {total_reward}")

# 清洗轨迹数据
valid_rewards = filter_abnormal_trajectory(all_rewards)
print(f"\n原始回合数: {len(all_rewards)} | 清洗后有效回合数: {len(valid_rewards)}")
print(f"原始轨迹总条数: {len(trajectory_data)}")

# 输出前3条清洗后的轨迹样本
print("\n=== 清洗后部分轨迹样本 ===")
for i in range(min(3, len(trajectory_data))):
    print(trajectory_data[i])

print("\nDRL 离线数据采集、标准化、异常过滤 全部完成")