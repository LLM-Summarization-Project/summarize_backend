import { Job } from "bullmq";
import path from "path";
import { spawn } from "child_process";
import axios from "axios";
import { PrismaClient } from '@prisma/client';
import * as fs from 'fs/promises';
import Redis from 'ioredis';

const prisma = new PrismaClient();
const ontology = process.env.ONTOLOGY_SERVICE;

// Redis for caching (same instance used by BullMQ)
const redis = new Redis({
    host: process.env.REDIS_HOST ?? 'localhost',
    port: Number(process.env.REDIS_PORT ?? 6379),
});

// Cache TTL: 1 month in seconds
const CACHE_TTL = 2592000;
const CACHE_PREFIX = 'summary:video:';

// Extract YouTube video ID from URL
function extractVideoId(url: string): string | null {
    const patterns = [
        /(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})/,
        /(?:youtu\.be\/)([a-zA-Z0-9_-]{11})/,
        /(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
        /(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/,  // YouTube Shorts
    ];
    for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match) return match[1];
    }
    return null;
}

// Cache completed summary
async function cacheSummary(youtubeUrl: string, summaryId: string) {
    try {
        const videoId = extractVideoId(youtubeUrl);
        const cacheKey = videoId
            ? `${CACHE_PREFIX}${videoId}`
            : `${CACHE_PREFIX}${Buffer.from(youtubeUrl).toString('base64').slice(0, 32)}`;

        await redis.setex(cacheKey, CACHE_TTL, JSON.stringify({
            id: summaryId,
            youtubeUrl,
            status: 'DONE',
        }));
        console.log(`[${summaryId}] Cached summary for ${cacheKey} (TTL: 1 month)`);
    } catch (error) {
        console.error(`[${summaryId}] Cache write error:`, error);
        // Don't throw - cache failure shouldn't break the job
    }
}

export async function processor(job: Job) {

    const { summaryId, youtubeUrl, userId } = job.data as {
        summaryId: string;
        youtubeUrl: string;
        userId: number;
    };

    let cancelledByUser = false;
    let cancelTimer: NodeJS.Timeout | null = null;

    // mark RUNNING
    await prisma.summary.update({
        where: { id: summaryId },
        data: { status: 'RUNNING', startedAt: new Date(), percent: 0 },
    });

    const user = userId.toString();
    const outputsDir = path.resolve(process.cwd(), 'outputs', user);
    const runnerPath = path.resolve(process.cwd(), 'python', 'runner.py');

    const py = spawn(
        process.env.PYTHON_BIN ?? 'python',
        [
            runnerPath,
            '--youtube_url',
            youtubeUrl,
            '--out_dir',
            outputsDir,
            '--scene_thresh',
            process.env.SCENE_THRESH ?? '0.4',
            '--language',
            process.env.LANGUAGE ?? 'th',
            '--asr_device',
            process.env.ASR_DEVICE ?? 'cpu',
            '--vl_device',
            process.env.VL_DEVICE ?? 'cpu',
            '--whisper_model',
            process.env.WHISPER_MODEL ?? 'large-v3-turbo',
            ...(process.env.OLLAMA_API
                ? ['--ollama_api', process.env.OLLAMA_API]
                : []),
            ...(process.env.OLLAMA_MODEL
                ? ['--ollama_model', process.env.OLLAMA_MODEL]
                : []),
            '--summary_id',
            summaryId,
        ],
        {
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                PYTHONIOENCODING: 'utf-8',
                PYTHONUTF8: '1',
            },
            cwd: process.cwd(),
            stdio: ['ignore', 'pipe', 'pipe', 'pipe'], // fd3 = progress
        },
    );

    if (!py.stdout || !py.stderr || !py.stdio[3]) {
        throw new Error('Python process missing stdio pipes');
    }

    py.stdout.setEncoding('utf-8');
    py.stderr.setEncoding('utf-8');

    let lastLine: string | null = null;
    let outBuf = '';
    let stderr = '';
    let asrPercent = 0;
    let modelLoaded = false; // track ว่าโหลดโมเดลเสร็จแล้วหรือยัง

    // 💡 helper: ดัก % จาก tqdm (เช่น " 27%|██▋       | 1552/5757 [...]")
    const handleTqdmChunk = async (chunk: string) => {
        // ถ้ายังโหลดโมเดลไม่เสร็จ ไม่ต้องจับ tqdm (เพราะอาจเป็น progress ของการโหลดโมเดล)
        if (!modelLoaded) return;

        const lines = chunk.split(/\r?\n/);
        for (const raw of lines) {
            const line = raw.trim();
            if (!line) continue;

            // จับตัวเลขก่อน % แล้วตามด้วย |
            const m = line.match(/(\d{1,3})%\s*\|/);
            if (!m) continue;

            const p = Number(m[1]);
            if (Number.isNaN(p)) continue;

            if (asrPercent === 100) return;
            const subprogress = Math.max(0, Math.min(100, p));
            asrPercent = subprogress;

            const percent = 10 + Math.floor((subprogress * 35) / 100); // tqdm ช่วง ASR = 10-45%

            await job.updateProgress({
                percent,
                step: 'ถอดเสียง',
                subprogress: subprogress,
            });

            await prisma.summary.update({
                where: { id: summaryId },
                data: { status: 'RUNNING', percent },
            });
        }
    };

    // STDOUT: เก็บบรรทัดสุดท้าย (เป็น JSON สรุป) + ดัก tqdm เผื่อมันพ่น stdout
    py.stdout.on('data', (chunk: string) => {
        outBuf += chunk;
        const lines = outBuf.split(/\r?\n/);
        outBuf = lines.pop() ?? '';
        for (const line of lines) {
            const t = line.trim();
            if (t) lastLine = t;
        }

        // ดัก % จาก tqdm ที่อาจโผล่ใน stdout
        void handleTqdmChunk(chunk);
    });

    // STDERR: เก็บ error log + ดัก tqdm (ส่วนใหญ่ tqdm อยู่ตรงนี้)
    py.stderr.on('data', (d: string) => {
        const text = d.toString();
        console.error(`[${summaryId}]`, text);
        stderr += text;

        // ดัก % จาก tqdm ที่พ่นบน stderr
        void handleTqdmChunk(text);
    });

    cancelTimer = setInterval(async () => {
        try {
            const s = await prisma.summary.findUnique({
                where: { id: summaryId },
                select: { status: true },
            });

            if (!s) return;
            if (s.status === 'CANCEL' && !cancelledByUser) {
                console.log(`[${summaryId}] detected CANCELLED in DB, sending SIGTERM to python`);
                cancelledByUser = true;
                py.kill('SIGTERM'); // หรือ 'SIGKILL' ถ้าอยากแบบแรง
                if (cancelTimer) {
                    clearInterval(cancelTimer);
                    cancelTimer = null;
                }
            }
        } catch (e) {
            console.error(`[${summaryId}] cancel check error`, e);
        }
    }, 1000);

    // FD3 = progress (JSON lines จาก pipeline)
    const progress = py.stdio[3] as NodeJS.ReadableStream;
    progress.setEncoding('utf8');
    progress.on('data', async (chunk: string) => {
        for (const line of chunk.split(/\r?\n/)) {
            const t = line.trim();
            if (!t) continue;
            try {
                const msg = JSON.parse(t);

                // จับ signal พิเศษว่าโหลดโมเดลเสร็จแล้ว
                if (msg?.type === 'model_loaded') {
                    modelLoaded = true;
                    console.log(`[${summaryId}] Whisper model loaded successfully`);
                    continue;
                }

                if (msg?.type === 'progress') {
                    const percent = Math.max(0, Math.min(99, Number(msg.percent) || 0));
                    await job.updateProgress({
                        percent,
                        step: msg.step ?? '',
                        subprogress: msg.subprogress ?? '',
                    });
                    await prisma.summary.update({
                        where: { id: summaryId },
                        data: { status: 'RUNNING', percent },
                    });
                }
            } catch {
                // ignore non-JSON
            }
        }
    });

    // Promise จบเมื่อโปรเซสปิด
    await new Promise<void>((resolve, reject) => {
        py.on('error', async (err) => {
            if (cancelTimer) {
                clearInterval(cancelTimer);
                cancelTimer = null;
            }

            await prisma.summary.update({
                where: { id: summaryId },
                data: {
                    status: 'ERROR',
                    errorMessage: `spawn error: ${err?.message ?? String(err)}`,
                    finishedAt: new Date(),
                },
            });
            reject(err);
        });

        py.on('close', async (code, signal) => {
            if (cancelTimer) {
                clearInterval(cancelTimer);
                cancelTimer = null;
            }

            // flush บรรทัดสุดท้าย
            if (outBuf.trim()) lastLine = outBuf.trim();

            // helper เดิมของคุณ
            const finishError = async (msg: string) => {
                await prisma.summary.update({
                    where: { id: summaryId },
                    data: {
                        status: 'ERROR',
                        errorMessage: msg,
                        finishedAt: new Date(),
                    },
                });
            };

            // 🟥 เคสนี้: user กด cancel → เราฆ่า python ไปเอง
            if (cancelledByUser || signal === 'SIGTERM' || signal === 'SIGKILL') {
                console.log(
                    `[${summaryId}] python exited due to cancel (code=${code}, signal=${signal})`,
                );

                // อัปเดต DB ให้เป็น CANCELLED
                await prisma.summary.update({
                    where: { id: summaryId },
                    data: {
                        status: 'CANCEL',
                        finishedAt: new Date(),
                        // จะเก็บ errorMessage ว่า "Cancelled by user" ก็ได้
                        errorMessage: 'Cancelled by user',
                    },
                });

                await job.updateProgress({
                    percent: 100,
                    step: 'ยกเลิกโดยผู้ใช้',
                    subprogress: 100,
                });

                // จะให้ BullMQ มองว่า "failed แบบพิเศษ" ก็ reject ด้วย error เฉพาะชื่อ
                return resolve();
            }

            // จากตรงนี้ลงไปคือ logic เดิมของคุณ (code === 0 / else)
            if (code === 0) {
                if (!lastLine) {
                    await finishError('Python exited 0 but no final JSON emitted.');
                    return reject(new Error('no-final-json'));
                }
                try {
                    const result = JSON.parse(lastLine);
                    const metrics = result.metrics;
                    if (result?.status && result.status !== 'ok') {
                        await finishError(
                            result?.errorMessage ?? 'Python returned error status.',
                        );
                        return reject(new Error('python-error-status'));
                    }

                    await prisma.summary.update({
                        where: { id: summaryId },
                        data: {
                            status: 'DONE',
                            percent: 100,
                            finishedAt: new Date(),

                            transcriptPath: result.transcript_path,
                            bulletPath: result.bullets_path,
                            summaryPath: result.article_path,
                            sceneFactsPath: result.scene_facts_path,

                            whisperModel: metrics.whisper_model,
                            whisperTemp: metrics.whisper_temp,
                            asrDevice: metrics.asr_device,
                            vlDevice: metrics.vl_device,
                            vlModel: metrics.vl_model,
                            sceneThresh: metrics.scene_thresh,
                            enableOcr: metrics.enable_ocr,

                            frames: metrics.frames,
                            bulletCount: metrics.bullets,
                            transcriptWord: metrics.transcript_words,
                            summaryWord: metrics.article_words,
                            keyword: metrics.keyword,
                            timeDownloadSec: metrics.t_download,
                            timeSpeechtoTextSec: metrics.t_asr,
                            timeCaptionImageSec: metrics.t_caption,
                            timeSummarizeSec: metrics.t_summarize,
                            timeTotal: metrics.t_total,
                            durationSec: metrics.duration_sec,
                        },
                    });
                    await job.updateProgress({
                        percent: 100,
                        step: 'บันทึกข้อมูล',
                        subprogress: 100,
                    });

                    let summaryContent: string | null = null;
                    if (result.article_path) {
                        try {
                            const normalizedPath = result.article_path.replace(/\\/g, '/');
                            const filepath = path.resolve(normalizedPath);
                            summaryContent = await fs.readFile(filepath, 'utf-8');
                        } catch (error) {
                            console.error(`Failed to read summary file:`, error);
                        }
                    }

                    try {
                        await axios.post(`${ontology}/ontology/topic`, {
                            userId,
                            name: metrics.keyword,
                            description: summaryContent,
                        })
                    } catch (e) {
                        console.log('Failed to call Ontology Service:', e.message)
                    }

                    // Cache the completed summary for future requests
                    await cacheSummary(youtubeUrl, summaryId);

                    resolve();
                } catch (e: any) {
                    await finishError(`Failed to parse final JSON: ${e?.message}`);
                    reject(e);
                }
            } else {
                if (lastLine) {
                    try {
                        const r = JSON.parse(lastLine);
                        if (r?.status === 'error') {
                            await finishError(r?.errorMessage ?? `exit ${code}`);
                            return reject(new Error(r?.errorMessage ?? `exit ${code}`));
                        }
                    } catch {
                        /* ignore */
                    }
                }
                await finishError(stderr || `exit ${code}`);
                reject(new Error(stderr || `exit ${code}`));
            }
        });
    });

    // สำเร็จ
    return true;
}