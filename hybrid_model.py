import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import json

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        d = self.decoder_fc(z).view(-1, 128, 8, 8)
        return self.decoder(d), mu, logvar

class HybridModel:
    def __init__(self,
                 config_path='hybrid_model_config.json',
                 classifier_path='best_model_efficientnet.pth',
                 vae_path='vae_anomaly.pth'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config_path) as f:
            config = json.load(f)
        self.threshold = config['threshold']
        self.class_names = config['class_names']

        self.classifier = models.efficientnet_b0(pretrained=False)
        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.classifier.features[4].parameters():
            param.requires_grad = True
        self.classifier.classifier[1] = nn.Linear(
            self.classifier.classifier[1].in_features, 4)
        checkpoint = torch.load(classifier_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.classifier.load_state_dict(checkpoint)
        self.classifier = self.classifier.to(self.device).eval()

        self.vae = VAE(latent_dim=64).to(self.device)
        self.vae.load_state_dict(
            torch.load(vae_path, map_location=self.device))
        self.vae.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def predict(self, pil_image):
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        img_vae = F.interpolate(img, size=(64,64), mode='bilinear', align_corners=False)

        with torch.no_grad():
            recon, _, _ = self.vae(img_vae)
            recon_error = ((recon - img_vae)**2).mean().item()

        if recon_error > self.threshold:
            return {'predicted_class': 'ANOMALY',
                    'confidence': round(recon_error, 6),
                    'reconstruction_error': round(recon_error, 6),
                    'threshold': round(self.threshold, 6),
                    'is_anomaly': True}

        with torch.no_grad():
            probs = torch.softmax(self.classifier(img), dim=1)
            conf, pred = probs.max(1)

        return {'predicted_class': self.class_names[pred.item()],
                'confidence': round(conf.item(), 4),
                'reconstruction_error': round(recon_error, 6),
                'threshold': round(self.threshold, 6),
                'is_anomaly': False}
