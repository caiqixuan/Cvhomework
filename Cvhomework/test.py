import argparse
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Data_and_Loader.WLFW_Dataset import WLFWDatasets
from model_structure.Backbone_net import BackboneNet

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34 # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def test(wlfw_val_dataloader, plfd_backbone):
    plfd_backbone.eval()

    nme_list = []
    cost_time = []
    with torch.no_grad():
        for img, landmark_gt, _, _ in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)

            start_time = time.time()
            _, landmarks = plfd_backbone(img)
            cost_time.append(time.time() - start_time)

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy() # landmark_gt

            if args.show_image:
                show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] * [112, 112]

                cv2.imwrite("xxx.jpg", show_img)
                img_clone = cv2.imread("xxx.jpg")

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255,0,0),-1)
                cv2.imshow("xx.jpg", img_clone)
                cv2.waitKey(0)

            nme_temp = compute_nme(landmarks, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # inference time
        print("inference_cost_time: {0:4f}s".format(np.mean(cost_time)))

def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    plfd_backbone = BackboneNet().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'],strict=False)

    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset, batch_size=1, shuffle=False, num_workers=0)

    test(wlfw_val_dataloader, plfd_backbone)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    # 预训练的断点模型保存的绝对路径
    parser.add_argument('--model_path', default="D:\First_Paper\Cvhomework\checkpoint\snapshot\checkpoint_epoch_100.pth.tar", type=str)
    # 测试数据集中的记录txt
    parser.add_argument('--test_dataset', default=r'D:\First_Paper\Cvhomework\data\test_data\list.txt', type=str)
    # 是否可视化
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
