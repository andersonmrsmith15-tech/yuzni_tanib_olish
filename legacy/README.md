# Yuz va tanani tanib olish tizimi

Kamera orqali odamlarni **yuz** va **tana** bo'yicha taniydigan, har safar yangi ko'rinishlardan
**o'rganib boradigan** lokal tizim. Barcha ma'lumotlar SQLite bazasida (`data/biometrics.db`) saqlanadi.
Hech narsa internetga yuborilmaydi.

## Ishga tushirish

```bash
./run.sh
```

Brauzerda oching: **http://127.0.0.1:5050** va kameraga ruxsat bering.

`run.sh` birinchi marta virtual muhit yaratadi, kutubxonalarni o'rnatadi va modellarni (~180 MB) yuklab oladi.

## Kirish va rollar

Har bir sahifa login talab qiladi. Birinchi marta ochganda **/signup** sahifasida hisob oching: **eng birinchi hisob
avtomatik admin bo'ladi**. Keyin ro'yxatdan o'tganlar "Foydalanuvchi" rolini oladi. Rolni admin **Hisoblar**
sahifasida o'zgartiradi.

| Rol | Ko'radigan sahifalar |
|---|---|
| **Admin** | Hammasi: Kamera, Ro'yxatdan o'tkazish, Foydalanuvchilar, Dashboard, Hisoblar, Sozlamalar |
| **Direktor** | Faqat Dashboard |
| **Foydalanuvchi** | Faqat Kamera (avto-o'rganishni o'chira olmaydi) |

Ruxsatlar serverda har bir so'rovda tekshiriladi: menyuda ko'rinmagan sahifa yoki API ga to'g'ridan-to'g'ri
murojaat qilinsa ham 403 qaytadi. Xavfsizlik choralari:

- parollar xeshlab saqlanadi (werkzeug scrypt/pbkdf2);
- 5 ta noto'g'ri urinishdan keyin login 5 daqiqaga bloklanadi;
- parol o'zgarganda boshqa qurilmalardagi sessiyalar bekor bo'ladi;
- oxirgi adminni o'chirib yoki uning rolini pasaytirib bo'lmaydi.

Admin **Sozlamalar** sahifasida Sign up ni yopib qo'yishi mumkin. Shunda yangi hisoblarni faqat admin qo'shadi.

## Bo'limlar

| Sahifa | Vazifasi |
|---|---|
| **Kamera** (`/`) | Jonli tanib olish. Yashil: tanildi, sariq: yangi ko'rinish o'rganildi, qizil: noma'lum, binafsha: tana |
| **Ro'yxatdan o'tkazish** (`/register`) | Ism, telefon, bo'lim va izoh kiritiladi. 7 bosqichli yo'riq bilan kameradan ~16 ta surat olinadi (old, chap, o'ng, yuqori, past, jilmayish, butun bo'y) yoki tayyor rasmlar yuklanadi. Bu odam allaqachon ro'yxatda bo'lsa, tizim ogohlantiradi |
| **Foydalanuvchilar** (`/users`) | Ma'lumotlarni tahrirlash, namunalarni ko'rish va o'chirish, yangi rasmlar bilan o'rgatish, foydalanuvchini o'chirish |
| **Hisoblar** (`/accounts`) | Faqat admin: hisob qo'shish, rol berish, bloklash, parolni tiklash, o'chirish |
| **Sozlamalar** (`/settings`) | Faqat admin: atribut chegaralari (soqol, ko'zoynak...), tanib olish chegaralari, Sign up ochiq yoki yopiq |
| **Dashboard** (`/dashboard`) | Ko'rsatkichlar, 14 kunlik va soatlik grafiklar, eng ko'p tanilganlar, yuz va tana ulushi, hodisalar jurnali |

## Qanday o'rganadi

1. Har bir odamda **yuz galereyasi** (128 o'lchamli SFace vektorlari) va **tana galereyasi**
   (768 o'lchamli ReID vektorlari) bor.
2. Kamerada odam **ishonchli** tanilsa (o'xshashlik ≥ 0.55, boshqa odamlardan farqi ≥ 0.10, yuz aniq va katta),
   va bu kadr galereyadagi namunalardan **yetarlicha farq qilsa** (boshqa burchak, yorug'lik, ko'zoynak, soqol),
   u yangi namuna sifatida saqlanadi.
3. Yuz orqali tanilgan odamning tanasi (qomati, kiyimi) tana galereyasiga qo'shiladi. Keyin yuzi ko'rinmasa ham
   tana bo'yicha **taxminan** taniladi.
4. Galereya to'lsa (yuz: 80, tana: 40), eng ko'p takrorlanadigan avtomatik namuna o'chiriladi.
   Qo'lda qo'shilgan namunalar o'chirilmaydi.
5. Yuklangan rasmlar tekshiriladi: yuz boshqa foydalanuvchiga ko'proq o'xshasa yoki shu odamga umuman o'xshamasa,
   rasm qabul qilinmaydi.

## Tashqi ko'rinish atributlari

Har bir yuz uchun quyidagilar aniqlanadi: 👓 ko'zoynak, 🕶️ qora ko'zoynak, 🧢 kepka yoki bosh kiyim, 🧔 soqol,
👨 mo'ylov, 😷 niqob, 🎧 quloqchin, 🧕 ro'mol, 💎 sirg'a. Natijalar:

- kamerada ism ostida va yon panelda ko'rinadi;
- bazaga yoziladi, shuning uchun profilda "odatiy ko'rinishi" (masalan, "Ko'zoynak: 80%") yig'iladi;
- Dashboardda "Bugun aniqlangan atributlar" bo'limida chiqadi.

Bu ish CLIP modeli bilan zero-shot usulida bajariladi: bosh qismining rasmi "bor" va "yo'q" matn tavsiflari bilan
solishtiriladi. Yangi atribut qo'shish yoki chegarani o'zgartirish uchun `app/attributes.py` dagi `ATTRS`
ro'yxatini tahrirlang. Chegarani admin **Sozlamalar** sahifasida ham o'zgartirishi mumkin. Kamera sahifasidagi
**Diagnostika** belgisi har bir atributning joriy ehtimolini chegara bilan birga ko'rsatadi. Natija bir necha kadr bo'yicha silliqlanadi, shuning uchun bir lahzalik xato ko'rinmaydi.

## Ma'lumotlar bazasi (`data/biometrics.db`)

- `users`: id, ism, telefon, bo'lim, izoh, yaratilgan vaqt, oxirgi ko'rilgan vaqt, necha marta ko'rilgani
- `samples`: yuz va tana namunalari. Har birida embedding (BLOB), JPEG kesimi (BLOB), manbasi
  (`register`, `upload` yoki `auto`) va vaqti saqlanadi
- `accounts`: tizim hisoblari (login, parol xeshi, rol, faol yoki bloklangan)
- `settings`: admin sozlamalari (chegaralar, Sign up holati)
- `events`: jurnal (`recognized`, `unknown`, `learn`, `register`, `upload`, `delete`). Tanib olish hodisasida aniqlangan atributlar ham saqlanadi

Zaxira nusxa olish uchun shu bitta faylni nusxalash kifoya (server to'xtatilgan holda).

## Sozlash

Chegaralar `app/engine.py` dagi `Config` klassida. Ularni ishlab turgan serverda ham o'zgartirsa bo'ladi:

```bash
curl -X POST localhost:5050/api/config -H 'Content-Type: application/json' -d '{"face_match": 0.45}'
```

## Cheklovlar

- Tana bo'yicha tanib olish kiyimga bog'liq. Odam kiyimini almashtirsa, yuzi qayta ko'ringuncha tana bo'yicha
  tanilmaydi. Shuning uchun tana natijasi "taxminiy" deb belgilanadi.
- Jonli ekanlikni tekshirish (liveness, anti-spoofing) yo'q: telefondagi rasm ham tanilishi mumkin.
  Kirishni nazorat qilish uchun qo'shimcha himoya kerak.
- Atributlar ehtimoliy: yupqa ramkali ko'zoynak, kalta soqol, yon tomondan yoki uzoqdan ko'rinish va yomon
  yorug'likda xato bo'lishi mumkin.
- Biometrik ma'lumotlar shaxsiy ma'lumot hisoblanadi. Odamlarni ro'yxatga olishdan oldin ularning roziligini oling.

## Modellar (OpenCV Zoo)

YuNet (yuz detektori), SFace (yuz tanib olish), YOLOX (odam detektori), YouTu ReID (tana tanib olish),
CLIP ViT-B/32 (atributlar).
