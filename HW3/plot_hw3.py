import pandas as pd
import matplotlib.pyplot as plt

PPOs = ["PPO1", "PPO2", "PPO3"]
DQNs = ["DQN1", "DQN2", "DQN3"]

PPO_ELs = []
PPO_ERMs = []
PPO_TLs = []

DQN_ERMs = []
DQN_TLs = []

for ppo in PPOs:
    df = pd.read_csv("csv/entropy_loss/" + ppo + ".csv")
    PPO_ELs.append(df)

for ppo in PPOs:
    df = pd.read_csv("csv/ep_rew_mean/" + ppo + ".csv")
    PPO_ERMs.append(df)

    steps = df["Step"]
    values = df["Value"]


for ppo in PPOs:
    df = pd.read_csv("csv/train_loss/" + ppo + ".csv")
    PPO_TLs.append(df)

for dqn in DQNs:
    df = pd.read_csv("csv/ep_rew_mean/" + dqn + ".csv")
    DQN_ERMs.append(df)

for dqn in DQNs:
    df = pd.read_csv("csv/train_loss/" + dqn + ".csv")
    DQN_TLs.append(df)


# Plot episode reward mean for PPOs
plt.figure()
for i in range(len(PPO_ERMs)):
    plt.plot(PPO_ERMs[i]["Step"], PPO_ERMs[i]["Value"], label=PPOs[i])
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean vs. Step for PPOs")
plt.legend()
plt.savefig("plots/ppo_ep_rew_mean.png")

# Plot entropy loss for PPOs
plt.figure()
for i in range(len(PPO_ELs)):
    plt.plot(PPO_ELs[i]["Step"], PPO_ELs[i]["Value"], label=PPOs[i])

plt.xlabel("Step")
plt.ylabel("Entropy Loss")
plt.title("Entropy Loss vs. Step for PPOs")
plt.legend()
plt.savefig("plots/ppo_entropy_loss.png")

# Plot episode reward mean for DQNs
plt.figure()
for i in range(len(DQN_ERMs)):
    plt.plot(DQN_ERMs[i]["Step"], DQN_ERMs[i]["Value"], label=DQNs[i])
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean vs. Step for DQNs")
plt.legend()
plt.savefig("plots/dqn_ep_rew_mean.png")

# Plot train loss for DQNs
plt.figure()
for i in range(len(DQN_TLs)):
    plt.plot(DQN_TLs[i]["Step"], DQN_TLs[i]["Value"], label=DQNs[i])
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Train Loss vs. Step for DQNs")
plt.legend()
plt.savefig("plots/dqn_train_loss.png")


# Plot episode reward mean for PPO1 and DQN1
plt.figure()
plt.plot(PPO_ERMs[0]["Step"], PPO_ERMs[0]["Value"], label="PPO1")
plt.plot(DQN_ERMs[0]["Step"], DQN_ERMs[0]["Value"], label="DQN1")
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean vs. Step for PPO1 and DQN1")
plt.legend()
plt.savefig("plots/ppo1_dqn1_ep_rew_mean.png")

# Plot episode reward mean for PPO2 and DQN2
plt.figure()
plt.plot(PPO_ERMs[1]["Step"], PPO_ERMs[1]["Value"], label="PPO2")
plt.plot(DQN_ERMs[1]["Step"], DQN_ERMs[1]["Value"], label="DQN2")
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean vs. Step for PPO2 and DQN2")
plt.legend()
plt.savefig("plots/ppo2_dqn2_ep_rew_mean.png")

# Plot episode reward mean for PPO3 and DQN3
plt.figure()
plt.plot(PPO_ERMs[2]["Step"], PPO_ERMs[2]["Value"], label="PPO3")
plt.plot(DQN_ERMs[2]["Step"], DQN_ERMs[2]["Value"], label="DQN3")
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Episode Reward Mean vs. Step for PPO3 and DQN3")
plt.legend()
plt.savefig("plots/ppo3_dqn3_ep_rew_mean.png")

# Plot train loss for PPO1 and DQN1
plt.figure()
plt.plot(PPO_TLs[0]["Step"], PPO_TLs[0]["Value"], label="PPO1")
plt.plot(DQN_TLs[0]["Step"], DQN_TLs[0]["Value"], label="DQN1")
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Train Loss vs. Step for PPO1 and DQN1")
plt.legend()
plt.savefig("plots/ppo1_dqn1_train_loss.png")

# Plot train loss for PPO2 and DQN2
plt.figure()
plt.plot(PPO_TLs[1]["Step"], PPO_TLs[1]["Value"], label="PPO2")
plt.plot(DQN_TLs[1]["Step"], DQN_TLs[1]["Value"], label="DQN2")
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Train Loss vs. Step for PPO2 and DQN2")
plt.legend()
plt.savefig("plots/ppo2_dqn2_train_loss.png")

# Plot train loss for PPO3 and DQN3
plt.figure()
plt.plot(PPO_TLs[2]["Step"], PPO_TLs[2]["Value"], label="PPO3")
plt.plot(DQN_TLs[2]["Step"], DQN_TLs[2]["Value"], label="DQN3")
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Train Loss vs. Step for PPO3 and DQN3")
plt.legend()
plt.savefig("plots/ppo3_dqn3_train_loss.png")

# Plot all train loss
plt.figure()
plt.plot(PPO_TLs[0]["Step"], PPO_TLs[0]["Value"], label="PPO1")
plt.plot(PPO_TLs[1]["Step"], PPO_TLs[1]["Value"], label="PPO2")
plt.plot(PPO_TLs[2]["Step"], PPO_TLs[2]["Value"], label="PPO3")
plt.plot(DQN_TLs[0]["Step"], DQN_TLs[0]["Value"], label="DQN1")
plt.plot(DQN_TLs[1]["Step"], DQN_TLs[1]["Value"], label="DQN2")
plt.plot(DQN_TLs[2]["Step"], DQN_TLs[2]["Value"], label="DQN3")
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Train Loss vs. Step for All Models")
plt.legend()
plt.savefig("plots/all_train_loss.png")
