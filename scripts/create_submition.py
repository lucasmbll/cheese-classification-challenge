import hydra
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
import pandas as pd
import torch
import scripts.ocr as ocr
import logging
import torch.nn.functional as F
logging.getLogger().setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

@hydra.main(config_path="configs/train", config_name="config", version_base=None)
def create_submission(cfg):
    logging.getLogger().setLevel(logging.ERROR)
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))
    reader = ocr.initialize_ocr(cfg.ocr_method)

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])
    ocr_identified = 0
    cheese_names = ocr.load_cheese_names(os.path.join(hydra.utils.get_original_cwd(),'cheese_ocr.txt'))

    nb_images = 0
    for i, batch in enumerate(test_loader):
        nb_images += len(batch[0])
        images, image_names = batch
        images = images.to(device)
        prob = model(images)
        prob = F.softmax(prob, 1)
        preds = prob.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        for j, image_name in enumerate(image_names):
            # Load the original image
            image_path = os.path.join(cfg.dataset.test_path, f"{image_name}.jpg")
            original_image = Image.open(image_path).convert('RGB')
            
            lab = ocr.classify_image(original_image, reader, cheese_names, cfg.threshold_ocr, increment=cfg.increment, ocr_method=cfg.ocr_method, comparison_method=cfg.comparison_method)
            if lab:
                if lab != preds[j] and prob[j].max()>0.2:
                    print(prob[j].max())
                    print(f"OCR detected label: {lab} whereas model predicted {preds[j]} for image {image_name}")
                preds[j] = lab
                ocr_identified += 1

        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)
    print(f"OCR identified {ocr_identified} labels")


if __name__ == "__main__":
    create_submission()
