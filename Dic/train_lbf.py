import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import random
import csv
import time

from utils_dt import load_config, create_save_directory, setup_logger, save_config, get_batch_ok, preprocess_data
from Networks.ReturnNet import ReturnNet
from Networks.TeammateEncoder import TeammateEncoder
from Networks.AdhocAgentEncoder import AdhocAgentEncoder
from Networks.GoalDecoder import GoalDecoder
from Networks.dt_models.decision_transformer import DecisionTransformer_lbf
from Data import CustomDataset
from Trainer import SequenceTrainer_lbf, BaseTrainer, GoalTrainer_lbf
from TestGame import Test
from Agent.Adhoc_DT import Adhoc_DT


# 定义训练函数
def train_model(logger, trainer, train_loader, val_loader, num_epochs, device, test_interval, save_interval, save_dir, K, act_dim, dt_train_steps, goal_steps, model_save_path="models"):
    start_time = time.time()
    # 测试类
    test = Test("LBF")
    
    for epoch in range(num_epochs):
        
        epoch_goal_loss = 0.0
        epoch_action_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, episodes_data in enumerate(pbar):
                
                loss_dt = trainer.train(episodes_data, train_steps=dt_train_steps, device=device, max_ep_len=episodes_data["state"].size(1), max_len=K, goal_steps=goal_steps)
                #loss_goal_dict = trainer.train(episodes_data, K, device, goal_steps)
                
                epoch_goal_loss += loss_dt["total_goal_loss"]
                epoch_action_loss += loss_dt["action_loss"]
                # 打印每个batch的损失
                pbar.set_postfix({
                    "Dt Loss": f"{loss_dt['action_loss']:.4f}",
                    "Goal Loss": f"{loss_dt['total_goal_loss']:.4f}",
                    "MIE Loss": f"{loss_dt['mie_loss']:.4f}",
                    "MSE Loss R": f"{loss_dt['mse_loss_r']:.4f}",
                    "MSE Loss G": f"{loss_dt['mse_loss_g']:.4f}"
                })

                # 日志记录
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}]\n "
                                f"Dt Loss: {loss_dt['action_loss']:.4f}\n "
                                f"Goal Loss: {loss_dt['total_goal_loss']:.4f}, MIE Loss: {loss_dt['mie_loss']:.4f}\n "
                                f"MSE Loss R: {loss_dt['mse_loss_r']:.4f}, BCE Loss G: {loss_dt['mse_loss_g']:.4f}")
            
        # 每个epoch结束后记录平均损失
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Goal Loss: {epoch_goal_loss / len(train_loader) :.4f}")
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Action Loss: {epoch_action_loss / len(train_loader) :.4f}")
        logger.info("===================================================================================")
        
        # 每个epoch结束后进行验证
        val_loss_dict = trainer.evaluate(val_loader, device=device, max_ep_len=next(iter(val_loader))["state"].size(1), max_len=K, goal_steps=goal_steps)
        # val_loss_dict = trainer_goal.evaluate(val_loader, device=device, goal_steps=goal_steps)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]\n"
        f"Action Validation Loss: {val_loss_dict['action_loss'] / len(val_loader):.4f} \n "
        f"Goal Validation Loss: {val_loss_dict['total_goal_loss'] / len(val_loader):.4f} \n "
        f"MIE Validation Loss: {val_loss_dict['mie_loss'] / len(val_loader):.4f} \n "
        f"MSE R Validation Loss: {val_loss_dict['mse_loss_r'] / len(val_loader):.4f} \n "
        f"BCE G Validation Loss: {val_loss_dict['mse_loss_g'] / len(val_loader):.4f}")
        logger.info("===================================================================================")

        # 将损失写入 CSV 文件
        val_csv_file_path = os.path.join(save_dir, 'val_loss.csv')
        with open(val_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, val_loss_dict['action_loss'] / len(val_loader), val_loss_dict['total_goal_loss'] / len(val_loader), epoch_action_loss / len(train_loader), epoch_goal_loss / len(train_loader), ])

        # 计算训练时间
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours, rem = divmod(epoch_duration, 3600)
        minutes, _ = divmod(rem, 60)
        logger.info(f"Completed in {int(hours)}h {int(minutes)}m")
        
        # 每隔指定的间隔进行测试
        if (epoch + 1) % test_interval == 0 or epoch + 1 == 1:
            agent = Adhoc_DT(
                dt_model=trainer.model, 
                state_encoder=trainer.adhocencoder, 
                return_net=trainer.returnnet, 
                goal_decoder=trainer.goaldecoder,
                env_type="LBF"
            )
            returns, var = test.test_game(100, agent, K)
            logger.info(f"{epoch + 1} Test Returns: {returns}")
            returns_csv_file_path = os.path.join(save_dir, 'test_returns.csv')
            with open(returns_csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, returns, var])
        

        # 每隔指定的间隔保存模型
        if (epoch + 1) % save_interval == 0 or epoch + 1 == 1:
            dir_path = os.path.join(save_dir, model_save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_path = os.path.join(dir_path, f"epoch_{epoch+1}.pth") 

        # 每隔指定的间隔保存模型
        if (epoch + 1) % save_interval == 0 or epoch + 1 == 1:
            dir_path = os.path.join(save_dir, model_save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_path = os.path.join(dir_path, f"epoch_{epoch+1}.pth") 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': {
                    'model': trainer.model.state_dict(),
                    'teamworkencoder': trainer.teammateencoder.state_dict(),
                    'adhocencoder': trainer.adhocencoder.state_dict(),
                    'returnnet': trainer.returnnet.state_dict(),
                    'goaldecoder': trainer.goaldecoder.state_dict(),
                },
            }, save_path)
            logger.info(f"Model checkpoint saved at {save_path}")


    end_time = time.time()
    total_duration = end_time - start_time
    total_hours, total_rem = divmod(total_duration, 3600)
    total_minutes, _ = divmod(total_rem, 60)
    print(f"Training completed in {int(total_hours)}h {int(total_minutes)}m")

if __name__ == "__main__":

    env = "LBF"
    config = load_config(f"./config/{env}_config.yaml")
    save_dir = create_save_directory()
    config["save_dir"] = save_dir
    logger = setup_logger(save_dir)
    
    # 训练设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    device = config["device"]
    # 移动模型到设备
    teammateencoder = TeammateEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"], num_heads=config["TeammateEncoder_num_heads"]).to(device)
    adhocagentEncoder = AdhocAgentEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"]).to(device)
    returnnet = ReturnNet(input_dim=config["embed_dim"]).to(device)
    goaldecoder = GoalDecoder(input_dim=config["embed_dim"], scalar_dim=1, hidden_dim=512, output_dim=config["state_dim"], num=config["num_agents"], state_dim=config["state_dim"]).to(device)
    dt = DecisionTransformer_lbf(
        state_dim=config["state_dim"],
        num_agents=config["num_agents"],
        act_dim=config["act_dim"],
        max_length=config["K"],
        max_ep_len=config["max_ep_len"],
        hidden_size=config['dt_embed_dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_inner=4*config['dt_embed_dim'],
        activation_function=config['dt_activation_function'],
        n_positions=1024,
        resid_pdrop=config['dt_dropout'],
        attn_pdrop=config['dt_dropout'],
        ).to(device)

    # DT 的 trainer准备
    warmup_steps = config['warmup_steps']
    optimizer = torch.optim.AdamW(
        list(teammateencoder.parameters()) + 
        list(adhocagentEncoder.parameters()) + 
        list(returnnet.parameters()) + 
        list(goaldecoder.parameters()) +
        list(dt.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # 初始化 Trainer
    trainer = SequenceTrainer_lbf(
        model=dt,
        teammateencoder=teammateencoder, 
        adhocencoder=adhocagentEncoder, 
        returnnet=returnnet, 
        goaldecoder=goaldecoder, 
        optimizer=optimizer,
        batch_size=config["batch_size"],
        get_batch=get_batch_ok,
        alpha=config["alpha"], 
        beta=config["beta"], 
        gama=config["gama"], 
        sigma=config["sigma"],
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: -torch.sum(a * torch.log(a_hat + 1e-9)) / a.size(0),
        eval_fns=None,
    )

    # 保存配置到输出文件夹
    save_config(config, save_dir)
    logger.info("Starting.")
    logger.info("Loading Data.")
    data_path = config["train_data_path"]

    # 加载数据
    load_start_time = time.time()
    data = torch.load(data_path)
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    hours, rem = divmod(load_duration, 3600)
    minutes, _ = divmod(rem, 60)
    logger.info(f"Data loaded in {hours} hours {minutes} minutes.")

    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    logger.info("Training Started.")
    # 开始训练
    train_model(logger, trainer,
                train_loader, val_loader,
                 num_epochs=config["num_epochs"], 
                 device=config["device"], 
                 test_interval=config["test_interval"],
                 save_interval=config["save_interval"], 
                 save_dir=save_dir, 
                 K=config["K"], 
                 act_dim=config["act_dim"], 
                 dt_train_steps=config["dt_train_steps"], 
                 goal_steps=config["goal_steps"],
                 model_save_path=config["model_save_path"])
                 
    # test(train_loader, device=config["device"], num_epochs=config["num_epochs"], batch_size=config["batch_size"])
    logger.info("Training completed.")