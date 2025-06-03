import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from models import TripletNet, FeatureExtractor, ResBlock  # If delete this line, the model loading will fail
from utils import TripletLoss, EarlyStopping, LRScheduler, TripletDataset, awgn, HDF5Loader, ChanIndSpecTransformer, SpecTransformer


def train_feature_extractor(args, dev_range=np.arange(0, 30, dtype=int), pkt_range=np.arange(0, 1000, dtype=int), snr_range=np.arange(20, 80)):
    """
    Train an RFF extractor using triplet loss.

    Args:
        args: The command line arguments.
        dev_range: The label range of LoRa devices to train the RFF extractor.
        pkt_range: The range of packets from each LoRa device to train the RFF extractor.
        snr_range: The SNR range used in data augmentation.

    """
    ##################################################
    # Load and preprocess the dataset
    ##################################################
    HDF5_loader = HDF5Loader()
    if args.data_transformer == "ChanIndSpec":
        converter = ChanIndSpecTransformer()
    elif args.data_transformer == "Spec":
        converter = SpecTransformer()

    # Load preamble IQ samples and labels.
    data, label = HDF5_loader.load_complex_IQ_samples(args.trainset_path, dev_range, pkt_range)

    # Add additive Gaussian noise to the IQ samples.
    data = awgn(data, snr_range)

    # Convert time-domain IQ samples to channel-independent spectrograms.
    data = converter.convert(data)

    ##################################################
    # Train prepare
    ##################################################
    model = TripletNet(in_channels=data.shape[-1], pool_size=(7, 7), pkt_desc_vec_len=512)
    triplet_loss = TripletLoss(alpha=args.margin)
    optimizer = RMSprop(params=model.parameters(), lr=1e-3)

    # Create callbacks during training. The training stops when validation loss
    # does not decrease for 20 epochs.
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0)
    lr_scheduler = LRScheduler(optimizer=optimizer, patience=10, factor=0.2)

    # Split the dasetset into validation and training sets.
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, label, test_size=0.1, shuffle=True)
    del data, label

    # Create the trainining generator.
    train_dataset = TripletDataset(data=train_data, labels=train_labels, device_range=dev_range)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Create the validation generator.
    valid_dataset = TripletDataset(data=valid_data, labels=valid_labels, device_range=dev_range)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    ##################################################
    # Train the model
    ##################################################
    model.to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        model.train()
        train_loss = 0.0
        for iteration, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(args.device)
            positive = positive.to(args.device)
            negative = negative.to(args.device)

            optimizer.zero_grad()
            anchor_desc_vec, positive_desc_vec, negative_desc_vec = model(anchor, positive, negative)
            loss = triplet_loss(anchor_desc_vec, positive_desc_vec, negative_desc_vec)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Train loss: {train_loss / (iteration + 1)}")

        model.eval()
        valid_loss = 0.0
        for iteration, (anchor, positive, negative) in enumerate(valid_loader):
            anchor = anchor.to(args.device)
            positive = positive.to(args.device)
            negative = negative.to(args.device)

            anchor_desc_vec, positive_desc_vec, negative_desc_vec = model(anchor, positive, negative)
            loss = triplet_loss(anchor_desc_vec, positive_desc_vec, negative_desc_vec)
            valid_loss += loss.item()

        print(f"Valid loss: {valid_loss / (iteration + 1)}")
        early_stopping(valid_loss)
        lr_scheduler(valid_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Save the model and its architecture.
    torch.save(model.feature_extractor, args.model_save_path)


def test_classification(
    args,
    dev_range_enrol=np.arange(30, 40, dtype=int),
    pkt_range_enrol=np.arange(0, 100, dtype=int),
    dev_range_clf=np.arange(30, 40, dtype=int),
    pkt_range_clf=np.arange(100, 200, dtype=int),
):
    """
    Performs a classification task and returns the classification accuracy.

    Args:
        args: The command line arguments.
        dev_range_enrol: The range of device labels used during enrollment.
        pkt_range_enrol: The range of packet labels used during enrollment.
        dev_range_clf: The range of device labels used during classification.
        pkt_range_clf: The range of packet labels used during classification.

    Returns:
        - label_pred: The predicted labels.
        - label_clf: The true labels.
        - acc: The classification accuracy.

    """
    ##################################################
    # Load the model and set it to evaluation mode
    ##################################################
    # Load the feature extractor.
    feature_extractor = torch.load(args.model_save_path).to(args.device)
    feature_extractor.eval()

    ##################################################
    # Load and preprocess the enrollment dataset
    ##################################################
    HDF5_loader = HDF5Loader()
    if args.data_transformer == "ChanIndSpec":
        converter = ChanIndSpecTransformer()
    elif args.data_transformer == "Spec":
        converter = SpecTransformer()

    # Load the enrollment dataset. (IQ samples and labels)
    enrol_data, enrol_labels = HDF5_loader.load_complex_IQ_samples(args.enrolset_path, dev_range_enrol, pkt_range_enrol)

    # Convert IQ samples to channel independent spectrograms. (enrollment data)
    enrol_data = converter.convert(enrol_data)
    enrol_data = torch.tensor(enrol_data, dtype=torch.float32).to(args.device)  # 转换为PyTorch张量并移动到GPU

    # # Visualize channel independent spectrogram
    # plt.figure()
    # sns.heatmap(data_enrol[0,:,:,0],xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
    # plt.gca().invert_yaxis()
    # plt.savefig('channel_ind_spectrogram.pdf')

    # Extract RFFs from channel independent spectrograms.
    enrol_feature = feature_extractor(enrol_data)
    del enrol_data

    ##################################################
    # Create a K-NN classifier using the RFFs extracted from the enrollment dataset.
    ##################################################
    knn_classifier = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    knn_classifier.fit(enrol_feature.cpu().detach().numpy(), np.ravel(enrol_labels))

    ##################################################
    # Load and preprocess the classification dataset
    ##################################################
    # Load the classification dataset. (IQ samples and labels)
    clf_data, clf_labels = HDF5_loader.load_complex_IQ_samples(args.clf_path, dev_range_clf, pkt_range_clf)

    # Convert IQ samples to channel independent spectrograms. (classification data)
    clf_data = converter.convert_to_spec(clf_data)
    clf_data = torch.tensor(clf_data, dtype=torch.float32).to(args.device)

    # Extract RFFs from channel independent spectrograms.
    clf_feature = feature_extractor(clf_data)
    del clf_data

    # Make prediction using the K-NN classifier.
    pred_labels = knn_classifier.predict(clf_feature.cpu().detach().numpy())

    # Calculate classification accuracy.
    acc = accuracy_score(clf_labels, pred_labels)
    print("Overall accuracy = %.4f" % acc)

    return pred_labels, clf_labels, acc


def test_rogue_device_detection(
    args,
    dev_range_enrol=np.arange(30, 40, dtype=int),
    pkt_range_enrol=np.arange(0, 100, dtype=int),
    dev_range_legitimate=np.arange(30, 40, dtype=int),
    pkt_range_legitimate=np.arange(100, 200, dtype=int),
    dev_range_rogue=np.arange(40, 45, dtype=int),
    pkt_range_rogue=np.arange(0, 100, dtype=int),
):
    """
    Performs the rogue device detection task using a specific RFF extractor.
    It returns false positive rate (FPR), true positive rate (TPR),
    area under the curve (AUC) and corresponding threshold settings.

    Args:
        args: The command line arguments.
        dev_range_enrol: The range of device labels used during enrollment.
        pkt_range_enrol: The range of packet labels used during enrollment.
        dev_range_legitimate: The range of device labels used during legitimate device detection.
        pkt_range_legitimate: The range of packet labels used during legitimate device detection.

    Returns:
        - FPR: The detection false positive rate.
        - TPR: The detection true positive rate.
        - ROC_AUC: The area under the ROC curve.
        - EER: The equal error rate.

    """

    def _compute_eer(fpr, tpr, thresholds):
        """
        Compute the equal error rate (EER) and the threshold to reach EER point.

        Args:
            fpr: The false positive rate.
            tpr: The true positive rate.
            thresholds: The threshold values.

        Returns:
            - eer: The equal error rate.
            - threshold: The threshold value to reach EER point.

        """
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))

        return eer, thresholds[min_index]

    ##################################################
    # Load the model and set it to evaluation mode
    ##################################################
    # Load the feature extractor.
    feature_extractor = torch.load(args.model_save_path).to(args.device)
    feature_extractor.eval()

    ##################################################
    # Load and preprocess the enrollment dataset
    ##################################################
    HDF5_loader = HDF5Loader()
    if args.data_transformer == "ChanIndSpec":
        converter = ChanIndSpecTransformer()
    elif args.data_transformer == "Spec":
        converter = SpecTransformer()

    # Load enrollment dataset.
    enrol_data, enrol_labels = HDF5_loader.load_complex_IQ_samples(args.enrolset_path, dev_range_enrol, pkt_range_enrol)

    # Convert IQ samples to channel independent spectrograms.
    enrol_data = converter.convert(enrol_data)

    enrol_data = torch.tensor(enrol_data, dtype=torch.float32).to(args.device)

    # Extract RFFs from cahnnel independent spectrograms.
    enrol_feature = feature_extractor(enrol_data)
    del enrol_data

    ##################################################
    # Create a K-NN classifier using the RFFs extracted from the enrollment dataset.
    ##################################################
    knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    knnclf.fit(enrol_feature.cpu().detach().numpy(), np.ravel(enrol_labels))

    ##################################################
    # Load and preprocess the legitimate dataset, rogue dataset
    ##################################################
    # Load the test dataset of legitimate devices.
    legitimate_data, legitimate_labels = HDF5_loader.load_complex_IQ_samples(args.legitset_path, dev_range_legitimate, pkt_range_legitimate)
    # Load the test dataset of rogue devices.
    rogue_data, rogue_labels = HDF5_loader.load_complex_IQ_samples(args.rogueset_path, dev_range_rogue, pkt_range_rogue)

    # Combine the above two datasets into one dataset containing both rogue
    # and legitimate devices.
    test_data = np.concatenate([legitimate_data, rogue_data])
    test_labels = np.concatenate([legitimate_labels, rogue_labels])

    # Convert IQ samples to channel independent spectrograms.
    test_data = converter.convert(test_data)

    test_data = torch.tensor(test_data, dtype=torch.float32).to(args.device)
    # Extract RFFs from channel independent spectrograms.
    test_feature = feature_extractor(test_data)
    del test_data

    # Find the nearest 15 neighbors in the RFF database and calculate the
    # distances to them.
    distances, indexes = knnclf.kneighbors(test_feature.cpu().detach().numpy())

    # Calculate the average distance to the nearest 15 neighbors.
    detection_score = distances.mean(axis=1)

    # Label the packets sent from legitimate devices as 1. The rest are sent by rogue devices
    # and are labeled as 0.
    true_labels = np.zeros([len(test_labels), 1])
    true_labels[(test_labels <= dev_range_legitimate[-1]) & (test_labels >= dev_range_legitimate[0])] = 1

    # Compute receiver operating characteristic (ROC).
    fpr, tpr, thresholds = roc_curve(true_labels, detection_score, pos_label=1)

    # The Euc. distance is used as the detection score. The lower the value,
    # the more similar it is. This is opposite with the probability or confidence
    # value used in scikit-learn roc_curve function. Therefore, we need to subtract
    # them from 1.
    fpr = 1 - fpr
    tpr = 1 - tpr

    # Compute EER.
    eer, _ = _compute_eer(fpr, tpr, thresholds)

    # Compute AUC.
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, eer


def args_parser():
    parser = argparse.ArgumentParser()
    ##################################################
    # For all modes.
    ##################################################
    parser.add_argument("--mode", type=str, choices=["Train", "Classification", "Rogue Device Detection"], default="Classification")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model_save_path", type=str, default="./models/feature_extractor.pt")
    parser.add_argument("--data_transformer", type=str, choices=["Spec", "ChanIndSpec"], default="Spec")

    ##################################################
    # For "Train" mode.
    ##################################################
    parser.add_argument("--trainset_path", type=str, default="./dataset/Train/dataset_training_aug.h5")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train the feature extractor.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training the feature extractor.")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for the triplet loss.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")

    ##################################################
    # For "Classification" mode.
    ##################################################
    parser.add_argument("--clf_path", type=str, default="./dataset/Test/channel_problem/A.h5")

    ##################################################
    # For "Rogue Device Detection" mode.
    ##################################################
    parser.add_argument("--legitset_path", type=str, default="./dataset/Test/dataset_residential.h5")
    parser.add_argument("--rogueset_path", type=str, default="./dataset/Test/dataset_rogue.h5")

    ##################################################
    # For both "Classification" and "Rogue Device Detection" mode.
    ##################################################
    parser.add_argument("--enrolset_path", type=str, default="./dataset/Test/dataset_residential.h5")

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = args_parser()

    if args.mode == "Train":
        train_feature_extractor(args, dev_range=np.arange(0, 30, dtype=int), pkt_range=np.arange(0, 1000, dtype=int), snr_range=np.arange(20, 80))

    elif args.mode == "Classification":
        # Specify the device index range for classification.
        test_dev_range = np.arange(30, 40, dtype=int)

        # Perform the classification task.
        label_pred, label_clf, acc = test_classification(
            args=args,
            dev_range_enrol=np.arange(30, 40, dtype=int),
            pkt_range_enrol=np.arange(0, 100, dtype=int),
            dev_range_clf=test_dev_range,
            pkt_range_clf=np.arange(100, 200, dtype=int),
        )

        # Plot the confusion matrix.
        conf_mat = confusion_matrix(label_clf, label_pred)
        classes = test_dev_range + 1

        plt.figure()
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted label", fontsize=20)
        plt.ylabel("True label", fontsize=20)

    elif args.mode == "Rogue Device Detection":

        # Perform rogue device detection task using three RFF extractors.
        fpr, tpr, roc_auc, eer = test_rogue_device_detection(
            args=args,
            dev_range_enrol=np.arange(30, 40, dtype=int),
            pkt_range_enrol=np.arange(0, 100, dtype=int),
            dev_range_legitimate=np.arange(30, 40, dtype=int),
            pkt_range_legitimate=np.arange(100, 200, dtype=int),
            dev_range_rogue=np.arange(40, 45, dtype=int),
            pkt_range_rogue=np.arange(0, 100, dtype=int),
        )

        # Plot the ROC curves.
        plt.figure(figsize=(4.8, 2.8))
        plt.xlim(-0.01, 1.02)
        plt.ylim(-0.01, 1.02)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label="Extractor 1, AUC = " + str(round(roc_auc, 3)) + ", EER = " + str(round(eer, 3)), color="r")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc=4)
        # plt.savefig('roc_curve.pdf',bbox_inches='tight')
        plt.show()
