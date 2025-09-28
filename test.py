# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader
# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_MDNANet import  Res_CBAM_block
from model.model_MDNANet import  MDNANet

class Trainer(object):  # 定义一个 Trainer 类，用于管理训练和测试流程
    def __init__(self, args):  # 初始化函数，接受外部传入的参数

        # Initial
        self.args  = args  # 保存输入的参数
        self.ROC = ROCMetric(1, args.ROC_bins)
        self.PD_FA = PD_FA(1, args.ROC_bins)  # 保持 bins 用于 np.zeros
        self.threshold = args.threshold  # 单独保存 threshold

        self.mIoU  = mIoU(1)  # 初始化 mIoU（平均交并比）评估指标
        self.save_prefix = '_'.join([args.model, args.dataset])  # 设置保存文件的前缀名
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)  # 加载模型参数，如通道数量和网络块数量

        # Read image index from TXT
        if args.mode == 'TXT':  # 如果模式为 'TXT'，从文本文件中加载数据集索引
            dataset_dir = args.root + '/' + args.dataset  # 数据集目录路径
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)  # 加载训练、验证和测试集的索引

        # Preprocess and load data
        input_transform = transforms.Compose([  # 定义数据预处理方法
                          transforms.ToTensor(),  # 转换为张量
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])  # 归一化图像，使用 ImageNet 均值和标准差
        testset = TestSetLoader(  # 初始化测试集加载器
            dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size,
            transform=input_transform, suffix=args.suffix)  # 设置数据路径、裁剪尺寸和后缀
        self.test_data = DataLoader(  # 使用 DataLoader 加载测试数据
            dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)  # 设置批量大小和工作线程数

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'MDNANet':  # 如果选择的模型是 'MDNANet'
            model = MDNANet(  # 初始化 DNANet 模型
                num_classes=1, input_channels=args.in_channels,
                block=Res_CBAM_block, num_blocks=num_blocks,
                nb_filter=nb_filter, deep_supervision=args.deep_supervision)  # 设置模型的通道数、块数和是否使用深度监督
        model = model.cuda()  # 将模型移动到 GPU 上
        model.apply(weights_init_xavier)  # 使用 Xavier 初始化权重
        print("Model Initializing")  # 打印初始化完成信息
        self.model = model  # 保存模型实例

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]  # 初始化最佳召回率记录
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]  # 初始化最佳精确率记录

        # Load trained model
        checkpoint = torch.load('result/' + args.model_dir)  # 加载预训练模型的检查点文件
        self.model.load_state_dict(checkpoint['state_dict'])  # 加载模型的权重

        # Test
        self.model.eval()  # 将模型设置为评估模式
        tbar = tqdm(self.test_data)  # 用 tqdm 包装测试数据加载器以显示进度条
        losses = AverageMeter()  # 用于记录平均损失
        with torch.no_grad():  # 在测试中禁用梯度计算（节省内存并加速）
            num = 0  # 初始化计数器
            for i, (data, labels) in enumerate(tbar):  # 遍历测试数据
                data = data.cuda()  # 将输入数据移动到 GPU
                labels = labels.cuda()  # 将标签移动到 GPU
                if args.deep_supervision == 'True':  # 如果启用深度监督
                    preds = self.model(data)  # 获取多个层的预测结果
                    loss = 0  # 初始化损失
                    for pred in preds:  # 遍历每层的预测结果
                        loss += SoftIoULoss(pred, labels)  # 累加 SoftIoU 损失
                    loss /= len(preds)  # 平均损失
                    pred = preds[-1]  # 使用最后一层的预测结果作为最终输出
                else:  # 如果未启用深度监督
                    pred = self.model(data)  # 获取模型预测
                    loss = SoftIoULoss(pred, labels)  # 计算单层的 SoftIoU 损失
                num += 1  # 更新计数器

                losses.update(loss.item(), pred.size(0))  # 更新损失统计
                self.ROC.update(pred, labels)  # 更新 ROC 指标
                self.mIoU.update(pred, labels)  # 更新 mIoU 指标
                self.PD_FA.update(pred, labels, threshold=self.threshold)


                true_positive_rate, false_positive_rate, recall, precision = self.ROC.get()  # 获取当前的评估指标
                _, mean_IOU = self.mIoU.get()  # 获取平均交并比
            FA, PD = self.PD_FA.get(len(val_img_ids))  # 获取探测率和虚警率

            # 输出获取的指标
            print(f"Mean IoU: {mean_IOU:.4f}")
            print(f"Detection Probability of Detection (PD): {PD[0]:.4f}")
            print(f"False Alarm Rate (FA): {FA[0]:.4f}")  # 输出虚警率

            scio.savemat(  # 保存评估结果为 .mat 文件
                dataset_dir + '/' + 'value_result' + '/' + args.st_model + '_PD_FA_' + str(255),
                {'number_record1': FA, 'number_record2': PD})

            save_result_for_test(dataset_dir, args.st_model, args.epochs, mean_IOU, recall, precision)  # 保存测试结果


def main(args):  # 定义主函数
    trainer = Trainer(args)  # 创建 Trainer 对象并执行初始化

if __name__ == "__main__":  # 检查是否以脚本形式运行
    args = parse_args()  # 解析命令行参数
    main(args)  # 调用主函数












