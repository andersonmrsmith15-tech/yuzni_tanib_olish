# Yuz va tanani tanib olish tizimi (Vercel + Neon + PWA)

Kamera orqali odamlarni **yuz** va **tana** bo'yicha taniydigan, har safar yangi ko'rinishlardan **o'rganib boradigan**
tizim. Ko'zoynak, kepka, soqol kabi atributlarni ham aniqlaydi. Rollar: admin, direktor va oddiy foydalanuvchi.
Telefonga ilova sifatida o'rnatiladi (PWA). Eski lokal versiya: [legacy/README.md](legacy/README.md).

## Arxitektura

```
 Telefon / kompyuter brauzeri                         Vercel                       Neon Postgres
 ┌───────────────────────────────┐   faqat vektorlar  ┌─────────────────────┐      ┌──────────────┐
 │ kamera → YuNet → SFace (yuz)  │ ─────────────────► │ api/index.py (Flask)│ ───► │ users        │
 │        → YOLOX → ReID (tana)  │  + kichik kesimlar │  login, rollar,     │      │ samples      │
 │        → CLIP (atributlar)    │ ◄───────────────── │  solishtirish,      │      │ events       │
 │ onnxruntime-web (WASM)        │   ism, o'rganish   │  o'rganish, jurnal  │      │ accounts ... │
 └───────────────────────────────┘                    └─────────────────────┘      └──────────────┘
```

- **Modellar brauzerda ishlaydi.** Video serverga yuborilmaydi. Serverga faqat 128 o'lchamli yuz va 768 o'lchamli
  tana vektorlari boradi, o'rganish va ro'yxatdan o'tishda esa kichik JPEG kesimlar ham yuboriladi.
- **Serverda og'ir kutubxona yo'q** (Flask, psycopg, numpy). Shuning uchun u Vercel serverless funksiyasiga sig'adi.
- **Brauzer natijalari eski OpenCV versiyasi bilan aynan bir xil.** Yuz vektorlarining o'xshashligi 1.0000
  (`tests/ml_parity.mjs`), shuning uchun lokal versiyadan ko'chirilgan namunalar o'zgarishsiz ishlaydi.
- **O'rganish xavfsiz.** Server "o'rganish mumkin" degan qarorni embeddingga bog'langan imzoli token bilan beradi,
  shuning uchun brauzer bazaga o'zboshimchalik bilan vektor qo'sha olmaydi.

| Model | Hajm | Qachon yuklanadi |
|---|---|---|
| YuNet + SFace (yuz) | 39 MB | Kamera yoqilganda (majburiy) |
| YOLOX + ReID int8 (tana) | 61 MB | "Tana" belgisi yoqilganda |
| CLIP ViT-B/32 int8 (atributlar) | 89 MB | "Atributlar" yoqilganda (Hugging Face'dan) |

Modellar bir marta yuklanadi va qurilmada keshlanadi.

## Vercelga deploy qilish

1. Vercel CLI ni o'rnating va tizimga kiring:
   ```bash
   npm i -g vercel
   vercel login
   ```
2. Loyiha papkasida deploy qiling:
   ```bash
   vercel            # birinchi marta: loyihani bog'lash (preview)
   ```
3. Vercel'da muhit o'zgaruvchilarini o'rnating (Project → Settings → Environment Variables yoki CLI):
   ```bash
   vercel env add DATABASE_URL production   # Neon pooled ulanish (.env dagi DATABASE_URL)
   vercel env add SECRET_KEY production     # uzun tasodifiy qiymat (.env dagi SECRET_KEY)
   vercel env add APP_TZ production         # Asia/Tashkent
   ```
4. Production'ga chiqaring:
   ```bash
   vercel --prod
   ```
5. Saytni oching. Hali hisob bo'lmasa, **/signup** da birinchi ochilgan hisob **admin** bo'ladi.

Kamera faqat HTTPS orqali ishlaydi. Vercel buni avtomatik ta'minlaydi, lokal sinovda esa `localhost` ham bo'ladi.

## Telefonga o'rnatish (PWA)

- **Android (Chrome):** saytni oching → ⋮ menyu → **"Ilovani o'rnatish"** (yoki "Bosh ekranga qo'shish").
- **iPhone (Safari):** saytni oching → "Ulashish" tugmasi → **"Bosh ekranga qo'shish"**.

Ilova o'z ikonkasi bilan to'liq ekranda ochiladi. Old va orqa kamerani 🔄 tugmasi bilan almashtirish mumkin.
Birinchi ochilishda modellar yuklanadi, buni Wi-Fi'da qilish tavsiya etiladi.

## Rollar

| Rol | Sahifalar |
|---|---|
| **Admin** | Kamera, Ro'yxatdan o'tkazish, Foydalanuvchilar, Dashboard, Hisoblar, Sozlamalar |
| **Direktor** | Faqat Dashboard |
| **Foydalanuvchi** | Faqat Kamera |

Ruxsatlar serverda har bir API so'rovida tekshiriladi. Qo'shimcha himoya:
- parollar xeshlab saqlanadi;
- 5 ta noto'g'ri urinishdan keyin login 5 daqiqaga bloklanadi;
- parol o'zgarganda boshqa qurilmalardagi sessiyalar bekor bo'ladi.

## Lokal ishga tushirish

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env              # DATABASE_URL va SECRET_KEY ni to'ldiring
.venv/bin/python scripts/migrate.py   # sxemani bazaga qo'llash
.venv/bin/python api/index.py         # http://127.0.0.1:5050
```

## Ma'lumotlar bazasi (Neon Postgres)

Sxema `lib/schema.sql` da saqlanadi va `scripts/migrate.py` bilan qo'llanadi (to'g'ridan-to'g'ri ulanish orqali).

- `users`: tanib olinadigan odamlar (ism, telefon, bo'lim, oxirgi atributlar)
- `samples`: yuz va tana vektorlari (BYTEA) va JPEG kesimlar
- `events`: jurnal (tanildi, noma'lum, o'rgandi, ro'yxat, kirish...)
- `accounts`, `login_attempts`: tizim hisoblari
- `settings`, `counters`: sozlamalar va galereya versiyasi

Eski lokal SQLite bazasini ko'chirish uchun: `python scripts/migrate_sqlite_to_neon.py legacy/data/biometrics.db`

Neon Free tarifida har bir branch uchun 1 GB joy bor. Bir odamning to'liq galereyasi taxminan 2 MB oladi
(80 ta yuz va 40 ta tana namunasi, rasmlari bilan), ya'ni bu ~400 odamga yetadi.

## Testlar

```bash
cd tests && npm install
python make_reference.py rasm1.jpg ...      # OpenCV natijalari (legacy modellari kerak)
npm test                                    # brauzer ML yadrosi va OpenCV mosligi
TEST_DATABASE_URL=... python api_e2e.py     # API: rollar, ro'yxat, tanish, o'rganish (FAQAT test branch!)
BASE=... FAKE_VIDEO=... node browser_e2e.mjs  # haqiqiy Chrome, soxta kamera bilan
```

## Papkalar

- `api/index.py`: Vercel funksiyasi
- `lib/`: server mantig'i (baza, rollar, tanib olish, o'rganish)
- `public/`: sahifalar, `js/ml.js` va `js/ml-core.js` (brauzer ML), `models/`, `vendor/ort/`, PWA fayllari
- `legacy/`: eski lokal versiya (OpenCV serveri). Vercelga deploy qilinmaydi

## Cheklovlar

- **Ishonch:** embeddinglar brauzerda hisoblanadi. Shuning uchun "user" hisobiga ega odam nazariy jihatdan soxta
  vektor yuborib tanilish natijasini o'zgartirishi mumkin. Kirishni qat'iy nazorat qilish (eshik ochish va h.k.)
  uchun liveness tekshiruvi va ishonchli qurilma kerak.
- **Telefonda tezlik:** haqiqiy telefonda hali o'lchanmagan. Kompyuterda yuz ~60 kadr/s, telefonda bundan ancha sekin
  bo'ladi. Tana va atributlar eng og'ir qism, eski telefonlarda ularni o'chirib qo'ying.
- **Tana bo'yicha tanish** kiyimga bog'liq va taxminiy hisoblanadi.
- **Rozilik:** biometrik ma'lumotlar Neon bulutida saqlanadi. Odamlarni ro'yxatga olishdan oldin roziligini oling.
