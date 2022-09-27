import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


from torch.autograd import Variable
from torchvision.utils import save_image


# Set random seed for reproducibility
manualSeed = 999

print("Random Seed: ", manualSeed)

data_root = "../data/celebA"
img_size = 64
batch_size = 128
lr = 0.0002
num_epochs = 50

feature_map_of_generator = 64
feature_map_of_discriminator = 64
num_of_channels = 3
latent_dim = 100

epoch_interval = 1


# Set random seed to random module and
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataset = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               # transforms.Resize([img_size, img_size]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_of_generator * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 8, feature_map_of_generator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 4, feature_map_of_generator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator * 2, feature_map_of_generator, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_generator),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_map_of_generator, num_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


generator = Generator().to(device)

generator.apply(weights_init)
print(generator)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_of_channels, feature_map_of_discriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator, feature_map_of_discriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 2, feature_map_of_discriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 4, feature_map_of_discriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_of_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_of_discriminator * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

adversarial_loss = nn.BCELoss()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated_imgs = generator(z)

        # train generator
        generator.zero_grad()

        expect = Variable(Tensor(generated_imgs.size(0), ).fill_(1.0), requires_grad=False)

        generated_loss = adversarial_loss(discriminator(generated_imgs).view(-1), expect)

        generated_loss.backward()

        optimizerG.step()

        # train discriminator with real_loss
        discriminator.zero_grad()

        batch_samples = data[0].to(device)

        output = discriminator(batch_samples).view(-1)

        valid = Variable(Tensor(output.size(0), ).fill_(1.0), requires_grad=False)

        real_loss = adversarial_loss(output, valid)

        # train discriminator with fake_loss
        output = discriminator(generated_imgs.detach()).view(-1)

        fake = Variable(Tensor(output.size(0), ).fill_(0.0), requires_grad=False)

        fake_loss = adversarial_loss(output, fake)

        discriminate_loss = (real_loss + fake_loss) / 2
        discriminate_loss.backward()

        optimizerD.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch+1, num_epochs, i, len(dataloader), discriminate_loss.item(), generated_loss.item())
        )

    if epoch % epoch_interval == 0:
        save_image(generated_imgs.data, "images/epoch_%d.png" % epoch+1, nrow=8, normalize=True)
