// Frame-accurate renderer: seeks the page's paused GSAP timeline frame by frame and pipes PNG screenshots into ffmpeg,
// so the output never depends on realtime playback. Adapted from the Piccaso film.
//
//   node render.cjs                                   -> actionman-research-film-draft.mp4 (full film, 1920x1080, 30 fps)
//   node render.cjs --stills 5,30.5,95                -> stills/still-<t>.png per timestamp
//   node render.cjs --from 87 --to 96 --out race.mp4  -> one section only (for iterating)
//   node render.cjs --cues cues.json                  -> dump the sound-effect cue list (window.CUES) for make_music.py
//   options: --size 1920x1080 --fps 30 --channel chrome|msedge|chromium --chrome <path to chrome.exe>
//
// The page must expose window.seek(t), window.DURATION and window.ready, and skip autoplay when ?render is set.
const fs = require('fs')
const path = require('path')
const { spawn } = require('child_process')
const { chromium } = require(process.env.PW_CORE || path.join(__dirname, 'node_modules', 'playwright-core'))

const arg = (name, def) => {
  const i = process.argv.indexOf(`--${name}`)
  return i > -1 ? process.argv[i + 1] : def
}
const [W, H] = arg('size', '1920x1080').split('x').map(Number)
const FPS = Number(arg('fps', 30))
const OUT = arg('out', 'actionman-research-film-draft.mp4')
const STILLS = arg('stills', null)
const CUES = arg('cues', null)
const PAGE = path.resolve(__dirname, arg('page', 'index.html'))
const WIN_CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

;(async () => {
  const exe = arg('chrome', fs.existsSync(WIN_CHROME) && !arg('channel', null) ? WIN_CHROME : null)
  const browser = await chromium.launch(exe ? { executablePath: exe } : { channel: arg('channel', 'chrome') })
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 })
  page.on('pageerror', e => console.error('page error:', e.message))
  await page.goto(require('url').pathToFileURL(PAGE).href + '?render')
  await page.evaluate(() => window.ready)
  await page.waitForTimeout(600)                      // font swap and image decode can land a frame after the promise

  if (CUES) {
    const cues = await page.evaluate(() => JSON.stringify(window.CUES))
    fs.writeFileSync(path.resolve(__dirname, CUES), cues)
    console.log(`wrote ${CUES}`)
  }
  if (STILLS) {
    fs.mkdirSync(path.join(__dirname, 'stills'), { recursive: true })
    for (const t of STILLS.split(',')) {
      await page.evaluate(t => window.seek(t), Number(t))
      await page.screenshot({ path: path.join(__dirname, 'stills', `still-${t}.png`), type: 'png' })
    }
    console.log(`wrote ${STILLS.split(',').length} stills to stills/`)
  }
  if (CUES || STILLS) { await browser.close(); return }

  const duration = await page.evaluate(() => window.DURATION)
  const from = Number(arg('from', 0)), to = Math.min(duration, Number(arg('to', duration)))
  // JPEG frames (q95) are ~3x faster to capture than PNG at 1080p and visually identical after x264
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', arg('preset', 'slow'), '-threads', arg('threads', '0'), '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path.resolve(__dirname, OUT)],
  { stdio: ['pipe', 'inherit', 'inherit'] })
  const f0 = Math.round(from * FPS), f1 = Math.round(to * FPS), t0 = Date.now()
  for (let f = f0; f < f1; f++) {
    await page.evaluate(t => window.seek(t), f / FPS)
    const buf = await page.screenshot({ type: 'jpeg', quality: 95 })
    if (!ff.stdin.write(buf)) await new Promise(resolve => ff.stdin.once('drain', resolve))
    if ((f - f0) % (FPS * 10) === 0) console.log(`frame ${f}/${f1}  ${((Date.now() - t0) / 1000).toFixed(0)} s`)
  }
  ff.stdin.end()
  await new Promise(resolve => ff.on('close', resolve))
  await browser.close()
  console.log(`wrote ${OUT} (${f1 - f0} frames @ ${FPS}fps, ${W}x${H})`)
})()
