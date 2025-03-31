import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd


def test(env, policy):
    policy.eval()
    state = env.reset()
    done = False
    schedule = []

    while not done:
        state_tensor = torch.FloatTensor(state).float()  # 确保为 FloatTensor
        action_probs, _ = policy(state_tensor)
        action = torch.argmax(action_probs).item()
        print(action)

        next_state, _, done = env.step(action)
        state = next_state

    # 生成调度结果
    for task_id in range(env.num_tasks):
        start_time = env.task_start_times[task_id]
        end_time = start_time + env.task_durations[task_id]
        machine = env.task_machines[task_id]  # 获取任务的机器分配信息
        schedule.append([task_id + 1, start_time, end_time,machine])  # 任务序号从1开始
        #schedule.append([task_id + 1, start_time, end_time])  # 任务序号从1开始

    return schedule


def plot_gantt(schedule, num_planes):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20.colors

    for i, task in enumerate(schedule):
        task_id, start_time, end_time, machine = task
        ax.barh(machine, end_time - start_time, left=start_time, color=colors[task_id % len(colors)])
        ax.text(start_time + (end_time - start_time) / 2, machine, f" {task_id}", va='center', ha='center')

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(num_planes))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(num_planes)])
    ax.set_title("Gantt Chart for Task Scheduling")
    plt.show()


class SchedulingEnv:
    def __init__(self):
        self.num_tasks = 36
        self.num_planes = 6
        # self.task_durations = [4, 4, 10, 2, 2, 5, 4, 5, 7, 3, 3, 3, 5, 9, 3, 2, 6, 3, 3, 3, 3, 3, 3]
        # self.pre_tasks = [[p - 1 for p in pre] for pre in [
        #     [], [1], [2], [3], [3], [4], [5], [6], [5, 8], [9], [1], [1], [8],
        #     [10, 12], [10], [10, 11], [16], [10], [10], [14, 17], [13], [21],
        #     [15, 18, 19, 20]
        self.task_durations = [5, 5, 5, 5, 5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 5, 5, 1, 1, 84, 2.5, 15, 8, 1, 5, 2, 134,134,9,13,92,67,7,20,15,15,15,10,10,10]
        self.pre_tasks = [[p - 1 for p in pre] for pre in [
            [], [], [], [], [], [23], [23], [23], [23], [23], [23], [0], [0],
            [23], [14], [23], [], [], [], [23], [23], [],
            [],[],[24],[],[35,36],[35,36],[35,36],[35,36],[35,36],[35,36],
            [],[],[34],[]
        ]]
        self.reset()

    def reset(self):
        self.task_completion = np.zeros(self.num_tasks, dtype=int)
        self.task_start_times = np.zeros(self.num_tasks)
        self.remaining_tasks = self.num_tasks
        self.machine_total_times = [0] * self.num_planes
        self.machine_work_time = [0] * self.num_planes
        self.total_completion_time = 0
        self.task_machines = [-1] * self.num_tasks
        return self.get_state()

    def get_state(self):
        task_completion = self.task_completion.astype(float)
        task_start_times = self.task_start_times   # 归一化
        remaining_tasks = np.array([self.remaining_tasks ])
        machine_times = np.array(self.machine_total_times)
        total_time = np.array([self.total_completion_time ])
        state = np.concatenate([task_start_times, remaining_tasks, machine_times, total_time])
        return state

    def step(self, action):
        available_tasks = []
        for task_id in range(self.num_tasks):
            if self.task_completion[task_id] == 0:
                if all(self.task_completion[p] == 1 for p in self.pre_tasks[task_id]):
                    available_tasks.append(task_id)
                    list(available_tasks)
                    #print(available_tasks)
                    #available_tasks = available_tasks
        if not available_tasks:
            done = self.remaining_tasks == 0
            reward = self._calculate_reward()
            return self.get_state(), reward, done

        durations = [self.task_durations[t] for t in available_tasks]

        # 计算所有可行任务的最早开始时间
        earliest_start_times = []
        for task_id in available_tasks:
            start_time = max(
                max([self.task_start_times[p] + self.task_durations[p] for p in self.pre_tasks[task_id]], default=0),
                min(self.machine_total_times)
            )
            earliest_start_times.append(start_time)

        if action == 5:
            task_idx = np.argmin(earliest_start_times)
        elif action == 4:
            task_idx = np.argmax(durations)
            # sorted_idx = np.argsort(durations)
            # task_idx = sorted_idx[-2] if len(durations) >= 2 else 0
        elif action == 2:
            task_idx = np.argmin(earliest_start_times)
        elif action == 3:
            task_idx = np.argmax(durations)
        elif action == 1:
            task_idx = np.argmin(earliest_start_times)
        elif action == 0:
            task_idx = np.argmax(durations)

            # task_idx = np.argmin(durations)
        selected_task = available_tasks[task_idx]
        duration = self.task_durations[selected_task]

        if action == 5:
            machine = np.argmin(self.machine_total_times)
            #machine = np.argmin(self.machine_total_times)
        elif action == 4:
            machine = np.argmin(self.machine_total_times)
            # sorted_machines = np.argsort(self.machine_total_times)
            # machine = sorted_machines[len(sorted_machines) // 2]
        elif action == 2:
            sorted_machines = np.argsort(self.machine_total_times)
            machine = sorted_machines[len(sorted_machines) // 2]
            # machine = np.argmax(self.machine_total_times)
        elif action == 3:
            sorted_machines = np.argsort(self.machine_total_times)
            machine = sorted_machines[len(sorted_machines) // 2]
        elif action == 1:
            machine = np.random.randint(0, self.num_planes)
        elif action == 0:
            machine = np.random.randint(0, self.num_planes)

        #start_time = self.machine_total_times[machine]
        self.task_machines[selected_task] = machine
        # start_time = max(self.machine_total_times[machine],
        #                  max([self.task_start_times[p] + self.task_durations[p] for p in self.pre_tasks[selected_task]],
        #                      default=0))
        start_time = max(self.machine_total_times[machine],
                         max([self.task_start_times[p] + self.task_durations[p] for p in self.pre_tasks[selected_task]],
                             default=0))
        #print(start_time)

        self.machine_total_times[machine] =  duration + start_time
        self.task_start_times[selected_task] = start_time
        self.machine_work_time[machine] += duration
        #print(self.machine_total_times)
        self.task_completion[selected_task] = 1
        self.remaining_tasks -= 1
        self.total_completion_time = max(self.machine_total_times)
        reward = self._calculate_reward()
        done = self.remaining_tasks == 0
        return self.get_state(), reward, done

    def _calculate_reward(self):
        completed_task_times = [
            self.task_durations[i] for i in range(self.num_tasks) if self.task_completion[i] == 1
        ]
        sum_duration = sum(completed_task_times)
        #sum_duration = sum(self.task_durations) #所有任务时间和
        std_dev = np.std(self.machine_total_times)
        makespan = self.total_completion_time #当前任务完成时间
        time_opt = sum_duration / (self.num_planes * makespan)# FTD
        sum_starts = sum(self.task_start_times) - sum_duration

        eta = 0.2
        offset_opt = sum_starts / ((self.num_tasks * makespan)*eta)

        mu1 = 0.05
        mu2 = 0.001
        reward = time_opt - mu2 * std_dev
        return reward

# PPO 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

# 加载模型
def load_model(model_path, state_dim, action_dim):
    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy


if __name__ == "__main__":
    env = SchedulingEnv()
    state_dim = len(env.get_state())
    action_dim = 6
    model_path = 'C:/Users/28123/Desktop/PPO_POLICY.pth'

    # 加载训练好的模型
    policy = load_model(model_path, state_dim, action_dim)

    # 执行调度任务
    schedule = test(env, policy)

    # 打印调度表
    df = pd.DataFrame(schedule, columns=["Task ID", "Start Time", "End Time", "Machine"])
    print("Schedule Table:")
    print(df)

    # 绘制甘特图
    plot_gantt(schedule, env.num_planes)