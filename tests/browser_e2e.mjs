// Haqiqiy Chrome'da (headless, soxta kamera bilan) to'liq sinov: login -> rasm bilan ro'yxatdan o'tkazish -> kamera.
// FAQAT test bazasiga ulangan lokal serverda ishga tushiring:
//   BASE=http://127.0.0.1:5098 FAKE_VIDEO=lena.y4m PHOTO=lena.jpg USER=tadmin PASS=adminpass1 OUT=./shots node browser_e2e.mjs
import fs from "node:fs";
import puppeteer from "puppeteer-core";

const { BASE, FAKE_VIDEO, PHOTO, USER, PASS, OUT = "." } = process.env;
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
fs.mkdirSync(OUT, { recursive: true });
let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "OK  " : "FAIL"} ${msg}`); if (!ok) fails++; };

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${FAKE_VIDEO}`,
         "--no-first-run", "--disable-features=Translate"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1366, height: 900 });
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
page.on("requestfailed", (r) => { if (!r.url().includes("/api/auth/status")) errors.push("requestfailed: " + r.url() + " " + r.failure()?.errorText); });

// 1) login
await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle0" });
check(page.url().includes("/login"), `kirilmagan holda /login ga yo'naltirildi (${new URL(page.url()).pathname})`);
await page.type("#username", USER); await page.type("#password", PASS);
await Promise.all([page.waitForNavigation({ waitUntil: "networkidle0" }), page.click("button[type=submit]")]);
check(new URL(page.url()).pathname === "/dashboard", "login -> next=/dashboard");
check(await page.evaluate(() => self.crossOriginIsolated), "crossOriginIsolated (ko'p oqimli WASM)");
await page.screenshot({ path: `${OUT}/dashboard.png`, fullPage: true });

// 2) rasm bilan ro'yxatdan o'tkazish (modellar brauzerda)
await page.goto(`${BASE}/register`, { waitUntil: "networkidle0" });
await page.click("#mUp");
await (await page.$("#files")).uploadFile(PHOTO);
await page.type("#name", "Lena Brauzer");
await page.type("#department", "Sinov");
let t = Date.now();
await page.click("#submit");
await page.waitForFunction(() => /ro'yxatdan o'tdi|allaqachon|alert bad/.test(document.querySelector("#msg").innerHTML), { timeout: 240000 });
if ((await page.$eval("#msg", (e) => e.innerText)).includes("allaqachon")) {
  check(true, "takroriy odam aniqlandi -> 'baribir qo'shish' bosildi");
  await page.click("#force");
  await page.waitForFunction(() => /ro'yxatdan o'tdi/.test(document.querySelector("#msg").innerHTML), { timeout: 60000 });
}
const msg = await page.$eval("#msg", (e) => e.innerText);
check(msg.includes("ro'yxatdan o'tdi"), `ro'yxat (${((Date.now() - t) / 1000).toFixed(1)} s): ${msg.split("\n")[0]}`);

// 3) kamera
await page.goto(`${BASE}/`, { waitUntil: "networkidle0" });
t = Date.now();
await page.click("#toggle");
try {
  await page.waitForFunction(() => document.querySelector("#present")?.innerText.includes("Lena Brauzer"), { timeout: 300000 });
  check(true, `kamerada tanildi (${((Date.now() - t) / 1000).toFixed(1)} s, modellar yuklanishi bilan)`);
} catch { check(false, "kamerada tanilmadi: " + (await page.$eval("#present", (e) => e.innerText))); }
try {
  await page.waitForFunction(() => document.querySelector("#present .chips")?.innerText.includes("bosh kiyim"), { timeout: 240000 });
  check(true, "atribut (shlyapa) brauzerda aniqlandi");
} catch { check(false, "atribut chiqmadi: " + (await page.$eval("#present", (e) => e.innerText))); }
await new Promise((r) => setTimeout(r, 1500));
check(true, "tezlik: " + (await page.$eval("#perf", (e) => e.innerText)));
await page.screenshot({ path: `${OUT}/camera.png` });

// 4) mobil ko'rinish
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
await new Promise((r) => setTimeout(r, 2500));
await page.screenshot({ path: `${OUT}/camera-mobile.png` });
await page.goto(`${BASE}/dashboard`, { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 1000));
await page.screenshot({ path: `${OUT}/dashboard-mobile.png`, fullPage: true });
const manifest = await page.evaluate(async () => (await fetch("/manifest.webmanifest")).ok);
const sw = await page.evaluate(async () => !!(await navigator.serviceWorker.getRegistration()));
check(manifest && sw, `PWA: manifest=${manifest}, service worker=${sw}`);

// login'dan oldingi /api/me 401 javobi kutilgan holat
const real = errors.filter((e) => !/favicon|status of 401/.test(e));
check(real.length === 0, `brauzer xatolari: ${real.length ? "\n  " + real.slice(0, 8).join("\n  ") : "yo'q"}`);
await browser.close();
console.log(fails ? `\n${fails} ta xato` : "\nHammasi o'tdi");
process.exit(fails ? 1 : 0);
