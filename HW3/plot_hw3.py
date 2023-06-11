#
# Created on Sat Jun 10 2023
#
# Deniz Karakay 2443307
#
# EE449 HW3 - Plotting
#


import pandas as pd
import matplotlib.pyplot as plt

PPOs = ["PPO1", "PPO2", "PPO3"]
DQNs = ["DQN1", "DQN2", "DQN3"]

best_name = ["PPO2_best"]

PPO_ELs = []
PPO_ERMs = []
PPO_TLs = []

DQN_ERMs = []
DQN_TLs = []


BEST_EL = pd.read_csv("csv/entropy_loss/" + best_name[0] + ".csv")
BEST_ERM = pd.read_csv("csv/ep_rew_mean/" + best_name[0] + ".csv")
BEST_TL = pd.read_csv("csv/train_loss/" + best_name[0] + ".csv")


for ppo in PPOs:
    df = pd.read_csv("csv/entropy_loss/" + ppo + ".csv")
    PPO_ELs.append(df)

for ppo in PPOs:
    df = pd.read_csv("csv/ep_rew_mean/" + ppo + ".csv")
    PPO_ERMs.append(df)

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


# Plot entropy loss for PPO1 and train loss for DQN1
plt.figure()
plt.plot(PPO_ELs[0]["Step"], PPO_ELs[0]["Value"], label="PPO1")
plt.plot(DQN_TLs[0]["Step"], DQN_TLs[0]["Value"], label="DQN1")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Entropy Loss of PPO1 and Train Loss of DQN1 vs. Step")
plt.legend()
plt.savefig("plots/ppo1_entropy_loss_dqn1_train_loss.png")

# Plot entropy loss for PPO2 and train loss for DQN2
plt.figure()
plt.plot(PPO_ELs[1]["Step"], PPO_ELs[1]["Value"], label="PPO2")
plt.plot(DQN_TLs[1]["Step"], DQN_TLs[1]["Value"], label="DQN2")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Entropy Loss of PPO2 and Train Loss of DQN2 vs. Step")
plt.legend()
plt.savefig("plots/ppo2_entropy_loss_dqn2_train_loss.png")


# Plot entropy loss for PPO3 and train loss for DQN3
plt.figure()
plt.plot(PPO_ELs[2]["Step"], PPO_ELs[2]["Value"], label="PPO3")
plt.plot(DQN_TLs[2]["Step"], DQN_TLs[2]["Value"], label="DQN3")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Entropy Loss of PPO3 and Train Loss of DQN3 vs. Step")
plt.legend()
plt.savefig("plots/ppo3_entropy_loss_dqn3_train_loss.png")

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

# Plot entropy loss for all PPO models and train loss for all DQN models
plt.figure()
plt.plot(PPO_ELs[0]["Step"], PPO_ELs[0]["Value"], label="PPO1")
plt.plot(PPO_ELs[1]["Step"], PPO_ELs[1]["Value"], label="PPO2")
plt.plot(PPO_ELs[2]["Step"], PPO_ELs[2]["Value"], label="PPO3")
plt.plot(DQN_TLs[0]["Step"], DQN_TLs[0]["Value"], label="DQN1")
plt.plot(DQN_TLs[1]["Step"], DQN_TLs[1]["Value"], label="DQN2")
plt.plot(DQN_TLs[2]["Step"], DQN_TLs[2]["Value"], label="DQN3")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Entropy Loss of PPO Models and Train Loss of DQN Models vs. Step")
plt.legend()
plt.savefig("plots/all_entropy_loss_and_train_loss.png")


# Plot best episode reward mean
plt.figure()
plt.plot(BEST_ERM["Step"], BEST_ERM["Value"], label="Best (PPO2)")
plt.xlabel("Step")
plt.ylabel("Episode Reward Mean")
plt.title("Best Episode Reward Mean vs. Step (PPO2)")
plt.legend()
plt.savefig("plots/best_ep_rew_mean.png")

# Plot best train loss
plt.figure()
plt.plot(BEST_TL["Step"], BEST_TL["Value"], label="Best (PPO2)")
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("Best Train Loss vs. Step (PPO2)")
plt.legend()
plt.savefig("plots/best_train_loss.png")

# Plot best entropy loss
plt.figure()
plt.plot(BEST_EL["Step"], BEST_EL["Value"], label="Best (PPO2)")
plt.xlabel("Step")
plt.ylabel("Entropy Loss")
plt.title("Best Entropy Loss vs. Step (PPO2)")
plt.legend()
plt.savefig("plots/best_entropy_loss.png")
