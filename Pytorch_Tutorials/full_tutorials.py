# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:26:57 2022

@author: 104863
"""

###############################################################################

# 1-) TENSORS(TENSÖRLER):
'''
Tensorler diziler ve matrislere çok benzeyen veri yapılarıdır. Modelin girdi ve çıktılarını ve modelin parametrelerini kodlamak için tensörleri kullanırız.
Tensörler numpy dizilerine çok benzer.
'''
import torch
import numpy as np

# Tensor oluşturma:
    
veri =  [[1,2], [3,4]]
tensor_veri = torch.tensor(veri) # direkt veriden.

np_dizi =  np.array(veri)
tensor_veri = torch.from_numpy(np_dizi) # numpy dizisinden.

tensor1 = torch.ones_like(tensor_veri) # başka bir tensörden.
print(f"Birinci Tensorümüz: {tensor1}")

random_tensor = torch.rand_like(tensor_veri,dtype=torch.float)
print(f"Random Tensörümüz: {random_tensor}")

shape = (2,3,) # shape ile herhangi bir boyutta..
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"random : {rand_tensor}")
print(f"ones : {ones_tensor}")
print(f"zeros : {zeros_tensor}")

# Tensör üzerinde kullanabileceğimiz metodlar:

tensor = torch.rand(3,4)

print(f"tensor'ün boyutu : {tensor.shape}")
print(f"tensor'ün tipi : {tensor.dtype}")
print(f"tensor hangi cihaz üzerinde çalışıyor, CPU/GPU:{tensor.device}")

# Tensör üzerindeki işlemler:
    
'''
Tensorler üzerinde aritmetik, mantıksal, lineer cebir, mantıksal, indexleme,
dilimleme gibi 100'den fazla işlemler yapılabilir. Biz bu bölümde en çok 
kullanılanlara bakacağız.(Detaylı bilgi için : https://pytorch.org/docs/stable/torch.html) 
Bu işlemler GPU da yüksek hızlarda çalışır.
Tensörler varsayılan olarak CPU da oluşturulur. Tensörü ilk etap da GPU' ya 
taşımak gerekir.
'''
# Tensor'ün GPU ya aktarılması:

if torch.cuda.is_available(): # eğer gpu ya geçersen.
    tensor = tensor.to('cuda') # tensor'u da gpu ya taşı
print('tensör gpu-ya taşındı..')

# Numpy benzeri indeksleme ve dilimleme:
tensor = torch.ones(4,4)
print(f"ilk satır: {tensor[0]}")
print(f" ilk kolon: {tensor[:,0]}")
print(f"son kolon: {tensor[...,-1]}")
tensor[:,1] = 0 # ones tensörünün birinci kolonundaki bütün verileri 0 yap.
print(tensor)

# tensörleri birleştirme:
t1 = torch.cat([tensor,tensor,tensor], dim=1) # dim=1 parametresi kolon bazında birleştirmek için , satır bazında birleştirmek için dim=0
print(t1)

# aritmetik işlemler:

# iki tensör arasında matrix çarpımı:
y1= tensor @ tensor.T
y2= tensor.matmul(tensor.T) # matmul = matrix multiplication
y3= torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out = y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3) # hesaplama.

# tensörün bütün elemanları toplayarak tek bir eleman haline getirme.
toplam = tensor.sum()
agg_item = toplam.item()
print(agg_item, type(agg_item))

# yerinde işlemler:
print(tensor, "\n")
tensor.add_(5) # tensör'ün bütün elemanlarına 5 ekle.
print(tensor)

# numpy ile köprü:
'''
CPU ve numpy dizilerindeki tensörler bellekte aynı konumları paylaşabilir.
Biri üzerinde yapılan değişiklik diğerini de etkileyebilir.
'''
t = torch.ones(5)
print(f"tensor: {t}")
n = t.numpy()
print(f"numpy: {n}")

# şuan cpu dayız. gpu ya geçmedik.
# tensördeki değişiklik numpy dizisine yansıyacaktır.
t.add_(1) # tensör elemanlarına 1 ekledik.
print(f"güncel tensör: {t}")
print(f"güncel numpy: {n}")

# aynı şekilde numpy'daki değişiklik tensöre de yansıyacaktır.
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n) # numpy dizisindeki elemanlara 1 ekle.
print(f"t: {t}")
print(f"n: {n}")

###############################################################################

# 2-) DATASETS & DATALOADERS:
'''
Veri kümesi işlemleri uzun ve zahmetlidir. Projelerde okunabilirlik ve modülerlik
için veri seti kodumuzu model eğitim kodumuzdan ayırırız.
Pytorch' da iki temel veri ilkesi vardır:
    1-) torch.utils.data.DataLoader
    2-) torch.utils.data.Dataset
Dataset: verileri ve bunlara karşılık gelen etiketleri saklar.
DataLoader: verilere kolay erişim sağlamamıza yardımcı olur.
2-) torch.utils.data.Dataset içerisinde paket olarak gelen bir dizi veri kümesi mevcuttur.
Biz bu eğitim setimiz içerisinde FashionMNIST veri kümesi üzerinden gideceğiz.
Daha detaylı bilgi için:
    Görüntü veri kümesi: https://pytorch.org/vision/stable/datasets.html
    Metin veri kümesi: https://pytorch.org/text/stable/datasets.html
    Ses veri kümesi: https://pytorch.org/audio/stable/datasets.html
'''  

# 1- veri kümesini yükleme:
''' 
FashionMNIST veri kümesi 60.000 eğitim ve 10.000 test örneğinden oluşur.
Her örnek 28*28 gri tonlamada 10 sınıftan birinden ilişkili bir etiket içeririr.
FashionMNIST veri kümesini aşağıdaki parametreler ile yüklüyoruz:
    
    root : tren/test verilerinin depolandığı yoldur,
    train : eğitim veya test veri kümesini belirtir,
    download=True : root adresinde veriler mevcut değilse, verileri internetten indirir.
    transform ve target_transform : etiket dönüşümlerini belirtin
'''
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data", # data isminde bir dosya oluşturulacak.
    train=True, # eğitim kümesi olduğu bilgisini verdik.
    download=True,# root dizininde böyle bir dosya olmadığı için internet üzerinden download edilecek.
    transform=ToTensor() # ve veriler tensor tipine çevrilecektir.
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# veri kümesini indexleme ve görselleştirme:
''' 
Veri kümemizi (dataset) liste gibi indexleyebiliriz.
'''
# labels_map isminde bir dic tanımladık.(indexleme için).
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# görselleştirme:
figure = plt.figure(figsize=(8, 8)) # figür tanımlaması yaptık.
cols, rows = 3, 3 # figür içerisinde resimler 3*3 lük matrix şeklinde görünecek.

# indexleme:
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # 1 boyutlu int türünde random rakam = sample_idx
    img, label = training_data[sample_idx] # label int türünde random rakam(labels_map içerisinde 0'dan 1'e indexlediğimiz rakam). img ise random 28*28 lik tek bir resim.
    # görselleştirme devam.
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# 2- özel veri kümesi yükleme:

'''
Özel bir veri kümesinin (Dataset) in üç işlevi kesinlikle uygulaması gerekir.
    __init__ , __len__ ve __getitem__
FashionMNIST veri kümesi incelendiğinde görüntülerin 'img_dir'
etiketleri(labels) ise CSV dosyası olan 'annotations_file' da saklanır.
'''

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # self.img_labels = pd.read_csv(annotations_file, names=['dosya ismi(veriler)', 'etiket dosyası(.csv olan)'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels) # veri kümemizdeki örnek sayısı.
              
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # resim yolunu oluşturduk. c://veriler/resim.jpg gibi
        image = read_image(img_path) # image değerine bu yolu atadık.
        label = self.img_labels.iloc[idx, 1] # label int türünde random rakam(labels_map içerisinde 0'dan 1'e indexlediğimiz rakam). 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label # son olarak image ve label döner.

# DataLoaders ile verilerimizi eğitim için hazırlama:
'''
Bir modeli eğitirken genellikle datesetimizi küçük gruplar halinde modele veririz.(multiprocessing)
DataLoader' lar tam da bu noktada bize yardımcı olur.
DataLoader' lar bize train_features ve train_labels özelliklerini döndürür.
'''

# DataLoader Kullanımı
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoader ile yüklenen verinin görüntüsünü ve etiketini gösterme: 
train_features, train_labels = next(iter(train_dataloader)) # train_features = random olarak 64 adet resmin dataloader hali. train_labels ise etiketleri.
print(f"Özellik: {train_features.size()}") # 64 tane 28*28 lik resim. bunlar 0-1 değerleri arasında tutulan point numberlar. 
print(f"Etiket: {train_labels.size()}")
img = train_features[0].squeeze() # 28*28 lik resim.
label = train_labels[0] # dic listesindeki etiket rakamı
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

###############################################################################

# 3-) TRANSFORMS:
'''
Veriler her zaman makine öğrenim algoritmalarına girecek kadar temiz ve düzenli gelmez.
Verileri eğitime uygun hale getirmek için uygulanan manipule işlemlerine dönüşüm diyoruz.
İki temel etiket vardır.
    transform:
    target_transform: 
FashionMNIST verisi PIL Image formatındadır. Ve etiketleri tam sayıdır.
Tensor dönüşümü yapmak için ToTensor ve Lambda kullanacağız.
ToTensor() :  Bir PIL görüntüsünü veya Numpy dizisindeki görüntülerin pixel yoğunluğu değerlerini 0 ile 1 aralığında ölçekler.
Lambda Dönüşümleri : Kullanıcı tanımlı herhangi bir işlevi uygular. 
'''
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # Burada tamsayıyı tensor e dönüştürmek için fn tanımladık.
    # İlk önce 10 boyutunda (veri kümemizdeki etiket sayısı) 0 tensör oluşturur ve etiket tarafından verilen dizine 
)

###############################################################################

# 4-) SİNİR AĞI OLUŞTURMA:
'''
Sinir ağları veriler üzerinde işlem yapan katmanlardan meydana gelir.
Torch üzerindeki nn modülü içerisinden çağrılır.
Pytorch' daki her modül nn.Module öğesinin alt sınıflarını oluşturur.
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# eğitim için CPU/GPU kararı:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Eğitimi İçin {device} kullanılacaktır.')

'''
Projelerimizde OOP(nesne tabanlı programlama) yapısı ile devam edeceğiz.
Pytorch bizi OOP kullanımına davet etmekte..
Hem ana sınıfımız olan (NeuralNetwork) için hem de nn.Module sınıfımız için __init__(yapıcılar) ekleyeceğiz.
Burada devreye super() yapıcısı girecek.
Sınıfımızın ana yapısı aşağıdaki gibi olur.
'''
# modelin oluşturulması ve eğitimi:
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # modele girecek inputları bir dizi şekline getirir.
        self.linear_relu_stack = nn.Sequential( # modelin inşası.
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), # buradaki 10 eğitim sonunda 10 adet labellerimıza denk gelmektedir.
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # modele eklediğimiz cihazları yazdırıp kontrol edelim.
print(model)

# tahmin
X = torch.rand(1, 28, 28, device=device)
logits = model(X) # linear_relu_stack' dan gelen 10 adet eğitilmiş sınıflarımız.
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Tahmin ettiğimi sınıf: {y_pred}")

# Model Katmanlarını Adım Adım Analiz Etme:

input_image = torch.rand(3,28,28) # 3 görüntüden oluşan rastgele örnekler alalım.
print(input_image.size())

# 1-) nn.Flatten :
'''
Her 2 boyutlu (2D) 28*28 lik görüntüyü 784 piksel değerinden oluşan diziye çevirir.
'''
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 2-) nn.Linear :
'''
Linear katman kayıtlı ağırlıkları ve tahminleri kullanarak veriyi linear bir eğride toplamaya çalışır.
'''
layer1 = nn.Linear(in_features=28*28, out_features=20) # aktivasyon fonksiyonuna girecek olan 20 input.
hidden1 = layer1(flat_image)
print(hidden1.size())

# 3-) nn.ReLU :
'''
Aktivasyon fonksiyonudur.
Yani sonuca gitmeli miyim , gitmemeli miyim diyen kişidir.
Birden fazla aktivasyon fonksiyonu mevcuttur. Linear , step, sigmoid vs. gibi.
Sihir burada! Magic toch.. Dolayısıyla matematik bir kere daha ön planda..
'''
print(f"ReLU'dan önce: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"ReLU'dan sonra: {hidden1}")

# 4-) nn.Sequential :
'''
Yukarıda adım adım verdiğimi işlemleri sıralı bir şekilde yapmamıza yardımcı olur.
'''
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10) # aktivasyon fonksiyonuna 20 giriş verdik. sonuç olarak 10 çıkış (labels) aldık.
)
input_image_son = torch.rand(3,28,28)
logits_son = seq_modules(input_image)

# 5-) nn.Softmax :
'''
Ağımızın son doğrusal katmanıdır.
Her sınıf için modelin tahmin ettiği değerleri 0 1 değerleri arasında ölçekler.dim=1 parametrelerin toplamını 1 olması gerektiğini bildirir.
'''
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits_son)

# 6-) Model parametrelerinin tamamını görmek için :

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

###############################################################################

# 5-) TORCH.AUTOGRAD KULLANIMI: (TORCH.AUTOGRAD | AUTOMATIC DIFFERENTIATION WITH / OTOMATİK FARKLILAŞMA )
'''
Sinir ağlarını eğitirken en sık kullanılan algoritma back propagation (geri yayılımdır.)
Bu algoritmada modele ait ağırlıklar loss (kayıp) fonksiyonunun gradyanına göre ayarlanır.
Bu gradyanı hesaplamak için Pytorch' da torch.autograd kullanırız.
y= ax+b linear fnk. düşünelim. w (a) ağırlık değerimizdir.
İyi bir öğrenme demek w ve b değerlerini optimize etmemiz demektir.
Bu nedenle, bu değişkenlere göre kayıp fonksiyonunun gradyanlarını hesaplayabilmemiz gerekir.
Bunu yapmak için, bu tensörlerin özelliklerini belirledik.requires_grad
'''
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # beklenen çıktı.
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b # tahmin sonrasında gelen değer. y_pred
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # minimize etmeye çalışıyoruz.

print('Tahmin için gradient fn  =', z.grad_fn) # bellekte bir nesne oluşturulur.
print('Kayıp için gradient fn =', loss.grad_fn)

'''
NOT: 
    Sinir ağındaki parametrelerin ağırlıklarını optimize etmek için
    parametrelere göre kayıp fonksiyonumuzun türevlerini hesaplamamız gerekiyor(loss.backward())
'''
# loss.backward() kullanımı:
'''
loss.backward() kullanımı performans açısından kötü sonuçlar doğurabilir. 
Dikkatli kullanılmalıdır.
Aynı grafik üzerinden birden fazla kullanım yapmamız gerekirse retain_graph=True parametresini vermemiz gerekecektir.
'''
loss.backward()
print(w.grad)
print(b.grad)

'''
NOT:
    Varsayılan olarak tüm tensörler requires_grad=True şeklindedir. Ve Gradyan hesaplamasını destekler.
    Ama diyelim ki modelimizi eğittik ve sonrasında kaydettik. Herhangi bir projede çağırdır.
    Elimizdeki veri setine uygulayacağız.
    Hesaplamayı torch.no_grad() bloguna alarak geriye dönük gradyan hesaplamasını durdurabiliriz.
        Bu işlemi yapma amacımız önceden eğitilmiş bir ağda ince ayar yapmak olabilir (frozen parameters.| donmuş parametreler.)
        Ya da hesaplamayı hızlandırmak için.
            
'''
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Aynı sonucu elde etmenin başka bir yolu, detach() yöntemini tensör üzerinde kullanmaktır:

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

###############################################################################

# 6-) MODEL PARAMETRELERİNİ OPTİMİZE ETME.
'''
Artık bir modelimiz ve verilerimiz olduğuna göre parametreleri modelimiz üzerinde 
optimize ederek modelimizi eğitme/doğrulama ve test etme zamanı.
Bir modeli eğitmek yenilemeli bir süreçtir.
Her yinelemede (epoch) model çıktı hakkında bir tahminde bulunur.
Tahmindeki hatayı hesaplar.
Parametrelerine göre hatanın türevlerini(gradyan iniş ) toplar. Ve optimize eder.
'''

# Önceki bölümlerde yazdığımız kodları toplayalım:
# Modelimizi oluşturalım : model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Hyperparameters( hiper parametreler ):
'''
Hiperparametreler, model optimizasyon sürecini kontrol etmenizi sağlayan ayarlanabilir parametrelerdir. 
Farklı hiperparametre değerleri, model eğitimini ve yakınsama oranlarını etkileyebilir.
    Epoch Sayısı - veri kümesi üzerinde yineleme sayısı,
    Batch Size - parametreler güncellenmeden önce ağ üzerinden yayılan veri örneklerinin sayısı,
    Öğrenme Oranı - model parametrelerinin her partide/dönemde ne kadar güncelleneceği. 
    Daha küçük değerler yavaş öğrenme hızı sağlarken, büyük değerler eğitim sırasında öngörülemeyen davranışlara neden olabilir.
'''
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimizasyon Döngüsü (Optimization Loop) : 
'''
Hiperparametrelerimizi ayarladıktan sonra, modelimizi bir optimizasyon döngüsü ile eğitebilir ve optimize edebiliriz.
Optimizasyon döngüsünün her yinelemesine epoch denir .
Her döngü iki ana bölümden oluşur:
Eğitim Döngüsü - eğitim veri kümesini yineleyin ve optimum parametrelere yakınsamaya çalışın.
Doğrulama/Test Döngüsü - model performansının iyileşip iyileşmediğini kontrol etmek için test veri kümesini yineleyin.
'''

# Kayıp Fonksiyonu (Loss Function):
'''
 Kayıp fonksiyonu , elde edilen sonucun hedef değere olan farklılığının derecesini ölçer 
 Eğitim sırasında en aza indirmek istediğimiz kayıp fonksiyonudur. 
 Kaybı hesaplamak için, verilen veri örneğinin girdilerini kullanarak bir tahmin yaparız.
 Ve bunu gerçek veri etiketi değeriyle karşılaştırırız.
 Yaygın kullanılanlar:
     nn.MSELoss, nn.NLLLoss,  nn.CrossEntropyLoss, nn.LogSoftmax, nn.NLLLoss.
'''
loss_fn = nn.CrossEntropyLoss()

# Optimizer:
'''
Optimizasyon, her eğitim adımında model hatasını azaltmak için model parametrelerinin ayarlanması sürecidir.
Optimizasyon algoritmaları , bu işlemin nasıl gerçekleştirildiğini tanımlar (bu örnekte Stokastik Gradyan İnişi kullanıyoruz). 
Tüm optimizasyon mantığı optimizer nesnesinde kapsüllenmiştir.
Burada SGD optimizer'ı kullanıyoruz; ayrıca, ADAM ve RMSProp gibi PyTorch'ta farklı türde modeller ve veriler için daha iyi çalışan birçok optimizer var.
Daha fazla bilgi için : https://pytorch.org/docs/stable/optim.html
'''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
Eğitim döngüsünün içinde optimizasyon üç adımda gerçekleşir:
    optimizer.zero_grad() 
    loss.backward().
    optimizer.step()
'''

# Optimizer'in tam kodu:

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # prediction ve loss hesaplama.
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# EĞİTİM: 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Eğitim tamamlandı..!")

###############################################################################

# 7-) MODEL KAYDETME / GERİ ÇAĞIRMA. 
import torch
import torchvision.models as models

'''
# model kaydı:
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
'''
# kaydedilen modeli yükleme:
'''
model = models.vgg16() 
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''

# ya da
torch.save(model, 'model.pth') # model kayıt.
#model = torch.load('model.pth') # model yükleme.
























