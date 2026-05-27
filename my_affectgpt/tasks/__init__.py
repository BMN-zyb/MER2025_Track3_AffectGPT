"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from my_affectgpt.common.registry import registry
from my_affectgpt.tasks.base_task import BaseTask
from my_affectgpt.tasks.video_text_pretrain import VideoTextPretrainTask

def setup_task(cfg):
    # 检查配置中是否包含任务名称配置项
    assert "task" in cfg.run_cfg, "Task name must be provided."
    # 从运行配置中获取任务名称（例如：video_text_pretrain）
    task_name = cfg.run_cfg.task
    # 通过注册表获取任务类并执行初始化配置
    # 1. registry.get_task_class() 根据任务名称查找已注册的任务类
    # 2. setup_task() 方法根据配置初始化具体任务实例
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    # 验证任务实例是否成功创建
    assert task is not None, "Task {} not properly registered.".format(task_name)
    return task  # 返回初始化完成的任务实例

__all__ = [
    "BaseTask",
    "VideoTextPretrainTask"
]
