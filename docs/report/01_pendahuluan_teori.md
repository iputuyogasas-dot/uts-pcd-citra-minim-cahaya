## 1. Pendahuluan

Foto yang diambil dalam cahaya minim memiliki luminans rendah, rentang kontras
sempit, dan pada ISO tinggi derau yang kuat. Mencerahkan citra secara langsung
ikut menguatkan derau dan dapat memudarkan atau menggeser warna, sehingga
pemilihan dan urutan operasi menentukan hasil.

Project ini membangun pipeline pengolahan citra **klasik** (tanpa deep learning
dan tanpa model pra-latih) untuk memperbaiki citra minim cahaya/berderau, lalu
mengukurnya terhadap citra acuan dari adegan yang sama dengan pencahayaan baik.
Tujuannya bukan skor tertinggi, melainkan keputusan rancangan yang didukung angka:

1. merancang pipeline ≥ 4 tahap yang sesuai karakteristik tiap kondisi;
2. mengimplementasikan ekualisasi histogram dan filter median dari nol (NumPy)
   dan membuktikan kesetaraannya dengan OpenCV;
3. membandingkan beberapa metode per tahap dengan PSNR, SSIM, dan ΔE2000
   serta uji Wilcoxon;
4. menganalisis kasus gagal dan mengusulkan perbaikan.

Seluruh parameter disetel hanya pada 6 skenario dev; angka akhir dilaporkan
hanya dari 4 skenario uji yang tidak pernah dipakai saat tuning.

## 2. Dasar teori singkat

**Koreksi gamma.** Pemetaan titik <i>s</i> = 255·(<i>r</i>/255)<sup>γ</sup>. Nilai
γ < 1 mencerahkan area gelap lebih kuat daripada area terang. *Gamma adaptif*
memilih γ per citra agar rata-rata luminans ternormalisasi μ menjadi target
<i>t</i>: γ = ln <i>t</i> / ln μ, sehingga μ<sup>γ</sup> = <i>t</i>.

**Ekualisasi histogram (HE).** Memetakan intensitas dengan fungsi distribusi
kumulatif (CDF) sehingga histogram keluaran mendekati seragam:
<i>s<sub>k</sub></i> = round((cdf(<i>k</i>) − cdf<sub>min</sub>) / (<i>N</i> − cdf<sub>min</sub>) · 255).
**CLAHE** melakukan HE per ubin dengan histogram yang dipotong pada *clip limit*
untuk membatasi penguatan derau, lalu menginterpolasi antar-ubin.

**Retinex (SSR/MSR).** Menganggap citra sebagai hasil kali iluminasi dan
reflektansi; iluminasi diperkirakan dengan blur Gaussian besar sehingga
<i>R</i> = log <i>I</i> − log(<i>G</i><sub>σ</sub> ∗ <i>I</i>). MSR merata-ratakan beberapa σ.

**Reduksi derau.** *Median* mengganti piksel dengan median tetangga <i>k</i>×<i>k</i>
(tahan derau impuls, menjaga tepi). *Gaussian* adalah rata-rata berbobot
(menghaluskan tepi). *Bilateral* menambah bobot kemiripan intensitas sehingga tepi
terjaga. *Non-local means* (NLM) merata-ratakan piksel yang lingkungannya mirip,
di mana pun letaknya.

**Unsharp mask.** <i>I</i>′ = <i>I</i> + <i>a</i>·(<i>I</i> − <i>G</i><sub>σ</sub> ∗ <i>I</i>)
menguatkan komponen frekuensi tinggi dengan kekuatan <i>a</i>.

**Ruang warna.** Perbaikan kontras dapat diterapkan pada kanal luminans (Y pada
YCrCb, L pada Lab, V pada HSV) agar rona tidak bergeser, atau pada tiap kanal RGB.
Karena Cr/Cb menyimpan *selisih* warna dalam satuan absolut, menaikkan Y saja
menurunkan saturasi relatif. Alternatifnya adalah mengalikan ketiga kanal RGB
dengan rasio <i>Y</i><sub>baru</sub>/<i>Y</i><sub>lama</sub>.

**Metrik.** PSNR = 10·log<sub>10</sub>(255²/MSE) mengukur galat piksel. SSIM
(Wang dkk., 2004) membandingkan luminans, kontras, dan struktur lokal (jendela
Gaussian σ = 1,5, rata-rata kanal RGB). ΔE2000 (CIEDE2000) mengukur beda warna
perseptual di ruang CIELAB; ΔE ≈ 2 nyaris tak terlihat, ΔE > 10 jelas berbeda.
**Uji Wilcoxon signed-rank** menguji apakah median selisih berpasangan dua metode
pada citra yang sama berbeda dari nol tanpa asumsi normalitas.
