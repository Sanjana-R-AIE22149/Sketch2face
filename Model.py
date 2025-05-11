import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataset import SketchFaceFusion
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
dataset=SketchFaceFusion(r"Data\resized_sketches",r"Data\resized_faces")
dataloader=DataLoader(dataset,batch_size=4, shuffle=True)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared(self.avg_pool(x))
        max_out = self.shared(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.spatial(torch.cat([avg_pool, max_pool], dim=1))
        x = x * spatial_out
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),  # downsample
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encode(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decode(x)

def show_tensor_image(tensor_img):
    img = tensor_img.detach().cpu()
    
    if img.dim() == 4:
        img = img[0]
    img = (img + 1) / 2
    
    img = img.permute(1, 2, 0).numpy()
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

class DualInputSketchFaceGenerator(nn.Module):
    def __init__(self):
        super(DualInputSketchFaceGenerator, self).__init__()
        
        self.sketch_encoder = nn.Sequential(
            EncoderBlock(3, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256)
        )
        self.edge_encoder = nn.Sequential(
            EncoderBlock(3, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256)
        )

        self.fusion_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.cbam = CBAM(512)

       
        self.res_blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        
        self.decoder = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, sketch, edge):
        sketch_feat = self.sketch_encoder(sketch)  # [B, 256, 32, 32]
        edge_feat = self.edge_encoder(edge)        # [B, 256, 32, 32]
        
        x = torch.cat([sketch_feat, edge_feat], dim=1)  # [B, 512, 32, 32]
        x = self.fusion_conv(x)
        x = self.cbam(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # Output: [B, 3, 256, 256]

        return x
    
class PatchDiscriminator(nn.Module):
    def __init__(self,in_channels=6):
        super(PatchDiscriminator,self).__init__()
        def conv_block(in_c,out_c,normalize=True):
            layers=[nn.Conv2d(in_c,out_c,4,stride=2,padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model = nn.Sequential(
            *conv_block(in_channels + 3, 64, normalize=False),  # +3 for face input
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # Output: Patch-level map
        )

    def forward(self, sketch_edge, face):
        x = torch.cat([sketch_edge, face], dim=1)  # [B, 9, H, W]
        return self.model(x)



D = PatchDiscriminator()

sketch_edge = torch.randn(1, 6, 256, 256)  # sketch + edge
face = torch.randn(1, 3, 256, 256)         # real or generated face

output = D(sketch_edge, face)
print(output.shape)  # Expected ~ [1, 1, 30, 30]

adversarial_criterion = nn.BCEWithLogitsLoss()
l1_criterion = nn.L1Loss()
import torch.optim as optim

generator = DualInputSketchFaceGenerator()
discriminator = PatchDiscriminator()

g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, device):
    generator.train()
    discriminator.train()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for sketch_edge, real_face in progress_bar:
        sketch_edge = sketch_edge.to(device)
        real_face = real_face.to(device)

       
        sketch, edge = sketch_edge[:, :3, :, :], sketch_edge[:, 3:, :, :]
        fake_face = generator(sketch, edge)


        real_labels = torch.ones_like(discriminator(sketch_edge, real_face))
        fake_labels = torch.zeros_like(discriminator(sketch_edge, fake_face.detach()))

        real_output = discriminator(sketch_edge, real_face)
        fake_output = discriminator(sketch_edge, fake_face.detach())

        d_real_loss = adversarial_criterion(real_output, real_labels)
        d_fake_loss = adversarial_criterion(fake_output, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) * 0.5

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        fake_output = discriminator(sketch_edge, fake_face)
        g_adv_loss = adversarial_criterion(fake_output, real_labels)
        g_l1_loss = l1_criterion(fake_face, real_face)

        g_loss = g_adv_loss + 100 * g_l1_loss  # 100 is L1 loss weight as in Pix2Pix

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        progress_bar.set_postfix({
            "D_loss": d_loss.item(),
            "G_loss": g_loss.item()
        })
        #print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    num_epochs=100
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, device)
        if epoch % 10 == 0:
            # Save model weights
            torch.save(generator.state_dict(), f"sketch2face_generator{epoch}.pth")
            
            # Save sample output image
            generator.eval()
            with torch.no_grad():
                sample_input = next(iter(dataloader))[0].to(device)  # assuming input is first item in batch
                sketch = sample_input[:, :3, :, :]
                edge = sample_input[:, 3:, :, :]
                fake_output = generator(sketch, edge)
                
                # Optionally, denormalize if needed (for display)
                # fake_output = (fake_output + 1) / 2

                save_image(fake_output, f"outputs/generated_epoch{epoch}.png", normalize=True)
            generator.train()
        print(f"Epoch {epoch} done!")
    #torch.save(generator.state_dict(),"sketch2face_generator100.pth")
    torch.save(generator,"sketch2face_generator_model100.pt")
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    def show_sample_outputs(generator, dataloader, device, num_samples=4):
        generator.eval()
        with torch.no_grad():
            for sketch_edge, real_face in dataloader:
                sketch_edge = sketch_edge.to(device)
                real_face = real_face.to(device)
                sketch = sketch_edge[:, :3, :, :]
                edge = sketch_edge[:, 3:, :, :]

                fake_face = generator(sketch, edge)

        
            def denorm(tensor):
                    return (tensor * 0.5 + 0.5).clamp(0, 1)

            sketch = denorm(sketch)
            edge = denorm(edge)
            fake_face = denorm(fake_face)
            real_face = denorm(real_face)

            # Show a few samples
            for i in range(min(num_samples, sketch.shape[0])):
                    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
                    axs[0].imshow(sketch[i].permute(1, 2, 0).cpu())
                    axs[0].set_title("Sketch")
                    axs[1].imshow(edge[i].permute(1, 2, 0).cpu())
                    axs[1].set_title("Edge")
                    axs[2].imshow(fake_face[i].permute(1, 2, 0).cpu())
                    axs[2].set_title("Generated Face")
                    axs[3].imshow(real_face[i].permute(1, 2, 0).cpu())
                    axs[3].set_title("Ground Truth")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                    break 

    show_sample_outputs(generator, dataloader, device)