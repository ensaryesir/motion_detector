import cv2, time, pandas
from datetime import datetime

hareketsiz_durum = None
hareket_listesi = [ None, None ] #Hareketli nesne göründüğünde listeye dök
time = [] #Hareket zamanı

data_frame = pandas.DataFrame(columns = ["Başlangıç", "Bitiş"]) #veri_cercevesi yüklenip, bir sütun başlatılıyor

video = cv2.VideoCapture(0) #videoyu yakalamak için VideoCapture nesnesi oluşturulur

while True:
    check, frame = video.read()

    hareket = 0 #Hareket başlatma - 0 : Hareket yok

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Renkli görüntüyü gri tonlamalı görüntüye dönüştürme / siyay-beyaz ayrımı kolay olduğu için
    gray = cv2.GaussianBlur(gray, (21, 21), 0) #Gri tonlamalı görüntüyü gauss bulanıklığına dönüştürme / değişimi kolayca bulmak için.
                                               # Imgproc.GaussianBlur(kaynakGoruntu, hedefGoruntu, new Size(100,100),0);

    if hareketsiz_durum is None:
        hareketsiz_durum = gray
        continue

    # OpenCV de arka plan temizleme işlemini absdiff metodu yapmaktadır.
    # Absdiff metodu verilen iki matris arasında çıkarma işlemi yapar bu çıkarma işlemi sonucunda değişen kısımlar yani hareketli kısımlar gösterilir.
    # Çıkarma işlemi sonucu mutlak değer olarak döndürülür.
    frame_farkı = cv2.absdiff(hareketsiz_durum, gray)

    # Hareketsiz durumumuz ile geçerli çerçeve arasındaki değişim 30'dan büyükse beyaz renk gösterir.
    # Thresholding: Giriş olarak verilen görüntüyü ikili görüntüye çevirmek için kullanılan bir yöntemdir. İkili görüntü (binary), görüntünün siyah ve beyaz olarak tanımlanmasıdır.
    thresh_frame = cv2.threshold(frame_farkı, 30, 255, cv2.THRESH_BINARY)[1] # Hassasiyet ayarları: 30'u arttır = hassasiyet azalır
                                                                                                   # 30'u azalt = hassasiyet artar

    # Bu operatör giriş olarak verilen görüntü üzerinde parametreler ile verilen alan içerisindeki sınırları genişletmektedir.
    # Bu genişletme sayesinde piksel gurupları büyür ve pikseller arası boşluklar küçülür.
    # OpenCV dilation operatörü için Imgproc içerisinde dilate() operatörü bulunmaktadır.
    # Bu metot parametre olarak giriş görüntüsü olacak bir mat nesnesi, çıkış görüntüsü için ikinci bir mat nesnesi ve yapısal element almaktadır.
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Hareketli nesnenin konturunu bulma
    # Konturlar aynı renk ve yoğunluğa sahip olan tüm kesintisiz noktaları sınır boyunca birleştiren bir eğri olarak basitçe açıklanabilir.
    # Konturlar şekil analizi,nesne algılama ve tanıma için çok yararlı bir araçtır.
    # Kontur bulunması istenirken daha doğru sonuç için binary(siyah-beyaz) formunda resim kullanılmalıdır.
    # FindContours yöntemiyle konturleri bulunan resim komple değişir orjinal halini bir daha kullanılamaz hale gelir. Bunun için resimi yazılımda yedeklemeniz gerekmektedir.
    # OpenCV'de kontur bulma işlemi siyah zeminde beyaz nesne bulmak gibidir. Bulunması gereken nesne beyaz arka plan siyah olmalıdır.
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cnts 2 değer döndürür.

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        # Hareketli nesnenin etrafında yeşil dikdörtgen oluşturma
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # En sondaki '3' yeşil çizgilerin kalınlığı

    hareket_listesi.append(hareket)
    hareket_listesi = hareket_listesi[-2:]

    # Hareketin başlangıç zamanını ekleme
    if hareket_listesi[-1] == 1 and hareket_listesi[-2] == 0:
        time.append(datetime.now())

    # Hareketin bitiş zamanını ekleme
    if hareket_listesi[-1] == 0 and hareket_listesi[-2] == 1:
        time.append(datetime.now())

    # cv2.imread görüntüyü yükler - cv2.imshow pencere oluşturur

    # Görüntüyü gri tonlamalı görüntüleme
    cv2.imshow("Gray Frame", gray)

    # Statik çerçeve için geçerli çerçeve farkını gösterme
    cv2.imshow("Difference Frame", frame_farkı)

    # Yoğunluk farkı 30'dan büyükse beyaz görüneceği siyah beyaz görüntüyü görüntüleme
    cv2.imshow("Threshold Frame", thresh_frame)

    # Nesnenin hareket konturu ile renk karesinin oynatılmasını görüntüleme
    cv2.imshow("Color Frame", frame)

    cikis = cv2.waitKey(1) # cv2.waitKey(1500) → 1.5sn ekranda görünür ve kapanır
    if cikis == ord('q'):
    # Kare içinde hareket varsa, hareketin bitiş zamanını ekler.
        if hareket == 1:
            time.append(datetime.now())
        break

# Veri çerçevesine hareket zamanı ekleme
for i in range(0, len(time), 2):
    data_frame = data_frame.append({"Başlangıç":time[i], "Bitiş":time[i + 1]}, ignore_index = True)

# Hareketlerin kaydedileceği bir CSV dosyası oluşturma
data_frame.to_csv("Hareket_Zamanları_Kayıtları.csv")

video.release()
cv2.destroyAllWindows()