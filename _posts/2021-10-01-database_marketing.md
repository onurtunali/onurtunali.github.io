---
layout: post
title: "Veritabanlı Pazarlama: Müşteri Analizi ve Yönetimi"
categories: ML
date:   2021-10-01 22:54:40 +0300
excerpt: "<b>Veritabanlı Pazarlama</b> (<i>Database Marketing</i>), <b>Müşteri İlişkileri Yönetimi</b> (<i>Customer Relationship Management</i>)'nin analitik ayağıdır. En kısa tanımı, müşteri veritabanını kullanarak pazarlama faaliyetlerinin üretkenliğini arttırmaktır."
---


*R. Blattberg, B. Kim ve S. Neslin*, 2008

* content
{:toc}

**Veritabanlı Pazarlama** (*Database Marketing*), **Müşteri İlişkileri Yönetimi** (*Customer Relationship Management*)'nin analitik ayağıdır. En kısa tanımı, müşteri veritabanını kullanarak pazarlama faaliyetlerinin üretkenliğini arttırmaktır. 

> **Not:** Bire bir çeviri "Veritabanı Pazarlama" anlam olarak veritabanının pazarlanmasına daha yakın olduğu için işlevi vurgulayan "Veritabanlı Pazarlama (VePa)" tercih edilmiştir. Bilişim terimlerinin Türkçe karşılıkları Türk Standartları Enstitüsünün 2006 tarihli "Bilişim Terimleri Sözlüğü" kitabı temel alarak seçilmiştir. Parantez içindeki (**K.N**) kişisel not anlamına gelir ve kitabın yazarlarından bağımsız iddiaları belirtir.


# I. Stratejik Zorluklar

# 1. Giriş

> 1. Veritabanlı Pazarlama nedir?
> 2. Veritabanlı Pazarlama neden daha önemli hale geliyor
> 3. Veritabanlı Pazarlama Süreci

Pazarlamanın en önemli üç faktörü yeni müşteri edinme, müşteriyi elde tutma ve müşteriyi geliştirmedir. Veritabanlı pazarlama (**VePa**) bu üç ayakla beraber müşteri veritabanını kullanarak pazarlama sürecinin üretkenliğini arttırmaya odaklanır. Daha sayısal ve analitk doğasından ötürü, alaylı doğrudan pazarlamanın tahsil görmüş kardeşi olarak da görülebilir.

Başka bir deyişle pazarlama yöntemlerini veya stratejisini müşterilerin sağladığı bilgi doğrultusunda ayarlamaktır. Klasik pazarlamadan farkı bu süreci büyük miktarda veriyi göz önüne alarak gerçekleştirmesidir.

Üç faktör daha ayrıntılı şekilde aşağıdaki gibi açıklanabilir:

- **Müşteri elde edilmesi:** yeni müşterilerin firmaya katılması
- **Müşterinin elde tutma:** cari müşterilerin firmayla iş yapmaya devam etmesi
- **Müşterinin geliştirmesi:** cari müşterilerin firmayla yaptığı ticaretin hacminin arttırılması

Müşteri verisinin analizi VePa'ya özgü bir süreç değildir. *CRM* ve doğrudan pazarlama da bu araçları kullanır. Fakat VePa'nın odak noktası budur. 

# 2. Neden Veritabanlı Pazarlama

> 1. Pazarlama üretkenliğini arttırmak
> 2. Müşteri ilişkisi oluşturmak ve iyileştirmek
> 3. Sürdürülebilir rekabet avantajı yaratmak

Pazarlama çabalarının üretkenliğini ölçmek için müşteri geridönüş oranlarının kar ve maliyet üzerindeki etkisi kullanılabilir. 1000 kişilik bir müşteri tabanı ve geridönüş oranlarına karışık gelen tablo aşağıdaki gibidir:

|Segment | Müşteri Sayısı | Geridönüş Oranı| Kar| Birikimli Kar |
|---|---|---|---|---|
| 1 | 200 | %4 | 390 |390 |
| 2 | 200 | %1 | 90 | 480 |
| 3 | 200 | %0,5 | 40 |520 |
| 4 | 200 | %0,1 | 0 | 520 |
| 5 | 200 | %0 | -100 |420 |

* Her müşteriye 0,5TL harcanır ve gelir 50TL'dir.
* Kar = (müşteri sayısı) x (geridönüş oranı) x (gelir) - 0,5 x (müşteri sayısı)

Basit bir örnek olmasına rağmen eğer **hedefleme** (*targeting*) 4. ve 5. segmenti hariç tutarak optimize edilirse birikimli kar azami seviyeye çıkarılabilir. Tabi ki geri dönüşüm oranlarının tahmini pazarlama girişiminden önce belirlenmelidir ve çok açık bir süreç değildir. Fakat müşteri veritabanı ve VePa yardımı ile tahmini modelleme daha hassas bir doğrulukla yapılabilir.

*CRM* taraftarlarının temel iddiası, güçlü müşteri ilişkilerinin marka  saadakatını arttırdığı yönündedir. Bunun altında **müşteri ömürboyu değeri (MÖD)** (*LTV*) ekonomik kavramı yatar. Yeni müşteri elde etme maliyeti ($a$) ve cari müşterileri elde tutma oranının ($r$) MÖD üzerindeki etkisi aşağıdaki matematiksel model ile nicel hale getirilir: 

$$ \text{Kar} = \ell(r,a) = N \left ( \sum_{t = 0}^{\infty} \frac{(R -c -m)r^{t}}{(1+\delta)^{t}} \right ) -Na \tag{2.1}$$

Sadece $r,a$ göz önünde bulundurularak Ceteris paribus analizi yapıldığında, kar fonksiyonun $r$ ile konveks  $a$ ile doğrusal ilişkisi olduğu görülebilir:

$$ = N (R -c -m) \left [ \sum_{t = 0}^{\infty} \left (\frac{r}{1+\delta} \right )^{t}  - \frac{a}{(R -c -m)} \right ]$$

Seri toplam incelenirse geometrik olduğu belirlenir,

$$0 < \frac{r}{1+\delta} < 1 \text{ geometrik seri } \sum_{i=0}^{\infty}r^{i} = \frac{1}{1-r}$$ 

Diğer terimler sabit değer altında paranteze alınırsa (2.1) sadeleştirilmiş haliyle aşağıdadır:

$$ \text{Kar} = \ell(r,a) = \frac{C_{1}}{1 + \delta -r} - \frac{a}{C_{2}} \tag{2.1}$$

Bu formül konveks testine tabi tutulduğunda ikinci türev sürekli pozitif olduğu için konveks fonksiyon olduğu kesindir:

$$ \frac{\partial \ell(r,a)}{\partial r} = \frac{1}{(1 + \delta -r)^{2}} > 0, \text{ konveks fonksiyon} $$

Daha yalın dille söylenirse çıkan sonuç müşteriyi elde tutmak, MÖD ölçüt alındığında, yeni müşteri kazanmaktan daha karlıdır. 

En son rekabet avantajı argümanının temellendirilmesi şöyledir. Müşteri informasyon dosyası (MİD) ve müşteri informasyon sistemi (MİS) her firmaya özgü olduğundan, çıkarılacak sonuçlar sektördeki her aktör için eşsiz olacaktır. Bu olguyu takip ederek, kusurlu hedeflemenin (müşteri segmentlerinin geri dönüşüm oranlarının tam belirlenememesi) standart olduğu durumlarda ki hemen hemen tüm reel sektörleri içine alır, VePa bilgi işlemede avantaj sağlar.

# 3. Veritabanlı Pazarlama için Organize Olmak

> 1. Müşteri merkezli organizasyon  
> 2. Veritabanlı Pazarlama stratejisi
> 3. Müşteri yönetimi: Müşteri merkezli organizasyonun yapısal temeli
> 4. İnformasyon yönetim süreci: Bilgi yönetimi
> 5. Maaş ve teşvikler
> 6. Personel

Pek çok firma ürün merkezli bir yapıya sahiptir. Buna karşılık,müşteri merkezli organizasyon VePa'nın çalışması için kilit yapılardan birisidir. 

| <img src="/img/dm/org.png"> |
| **Fig 1:** Müşteri merkezli organizasyonun yıldız yapısı |

Firmanın kaynakları açısından değerlendirilmesi 4 faktörle yapılabilir:

1. Heterojenlik: Farklı departmanlar diğerlerinden daha verimlik veya üretken olması
2. Mevcut limit: Firmanın yeterliliklerini taklit etmenin güç olması
3. Kusurlu hareketlilik: Firmanın kaynaklarının kolayca tedarik edilemesi veya personel transferinin alt yapı eksikliğinden ötürü verimli olmaması
4. İlk olma avantajı: Erken benimsemeden ötürü kazanılmış üstünlük firma performansına yetişmeyi bazen imkansız kılması 

# 4. Müşterinin Gizliliği ve Veritabanlı Pazarlama

> 1. Tarihi perspektif
> 2. Gizliliğe karşı müşteri tavrı
> 3. Gizlilikle ilgili cari uygulamalar
> 4. Gizlilik endişeleri için potansiyel çözümler

Özellikle sanal dünyayı kapsayan gizlilikle ilgili düzenlemeler, Türkiye'de uzun bir geçmişe sahip değildir. Kişisel Verileri Koruma Kanunu (KVKK) 2016 yılında yürürlüğe girmiştir.

Müşterilerin gizlilikle ilgili endişeleri aşağıdaki gibi sıralanabilir:

- Veri gizliliği
- İzinsiz ve gizlice veri toplanması
- İstenmeyen (spam) email
- Verilerin üçüncü şahıslar veya kurumlarla paylaşılması
- Özel hayatın gizliğinin ihlal edilmesi

Bu endişeler ve VePa'nın veriye olan ihtiyacı temel çatışma noktasını oluşturur. Gizlilik endişeleri açısından müşteriler üç kategoriye ayrılabilir: muhafazakar, endişe seviyesi yüksek ve hiçbir koşul altında verilerinin kullanılmasını istemeyenler; faydacı, endişe seviyesi sınırlı ve tatmin edici düzenlemelerle verilerinin kullanılmasına karşı çıkmayanlar; kayıtsız, endişe seviyesi çok düşük ve sınırlı. Kategorilerin dağılımı sırasıyla %17, %56 ve %27'dir. 

Bahsedilen çatışmayı çözmek için önerilen çözümler 2 eksen etrafında toplanır: yazılım ve yönetmelik odaklı. Yazılımsal çözümler genelde hassas bilgilerin anonim hale getirilmesi işlemi üstünde durur. Bu şekilde sadece sınırlı bir yönetim kadrosu gerçek bilgilerle anonim bilgilerin karşılaştırabilir. Yönetmelik çözümleri ise müşteri ve firma arasındaki ilişkiyi, izin verilen eylemleri ve iki tarafın haklarını açık bir şekilde tanımlayarak ihlalleri asgari seviyeye çekmeye çalışır. 

Temel sonuçları tekrarlarsak (**K.N:** Burada kitabın biraz eskimiş, 2008 yılında yayımlanmış, olduğu ortaya çıkıyor. Özellikle son yıllardaki sosyal medya veri toplama ve sızma skandallarından sonra son madde aşikar bir olgu haline gelmiştir):

- Gizlilik çok boyutludur ve indirgeyici yaklaşımla çözülemez.
- Müşterilerin gizliliğe karşı olumsu tavrı satışları azaltır.
- Firmalar gizlilik politikalarını müşterilerine aktarır.
- Müşteri verileri için aktif bir piyasa vardır.

# 5. Müşteri Ömürboyu Değeri (MÖD): Temeller

> 1. Giriş
> 2. MÖD'ün matematiksel formülasyonu
> 3. İki temel MÖD modeli: Basit elde tutma ve oynak
> 4. Gözlemlenmeyen müşteri ayrılışını dikkate alan MÖD modeli
> 5. Gelir tahmini

Pazarlama çabaları, performansının nicel bir şekilde ölçülmesi ve yönetim kadrosunu ürettiği değer açısından ikna etmesi için anahtar ölçütlere (*metrics*) ihtiyaç duyar. Toplam satış kalemleri kuş bakışı bir görünüm sunduğu için daha ayrıntılı analizi mümkün kılan MÖD kavramı ileri sürülmüştür.



