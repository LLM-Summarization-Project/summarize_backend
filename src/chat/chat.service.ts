import { Injectable } from '@nestjs/common';
import { CreateChatDto } from './dto/create-chat.dto';
import axios from 'axios';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from 'src/summarize/prisma/prisma.service';
import path from 'path';
import * as fs from 'fs/promises';
import { Observable, Subject } from 'rxjs';

@Injectable()
export class ChatService {
  private ollamaApiUrl: string;
  private ollamaModel: string;
  
  constructor(configService: ConfigService, private prisma: PrismaService) {
    this.ollamaApiUrl = configService.get<string>('OLLAMA_API_CHAT') || 'http://localhost:11434/api/chat';
    this.ollamaModel = configService.get<string>('OLLAMA_MODEL') || 'scb10x/llama3.1-typhoon2-8b-instruct';
  }

  // Streaming version สำหรับ SSE
  async createStream(dto: CreateChatDto): Promise<Observable<MessageEvent>> {
    const summary = await this.prisma.summary.findUnique({ where: {id: dto.summaryId}})
    if(!summary) {
      throw new Error('Summary not found');
    }

    const normalizedPath = summary.transcriptPath?.replace(/\\/g, '/');
    if (!normalizedPath) {
      throw new Error('Summary file not found');
    }
    
    const filepath = path.resolve(normalizedPath || '');
    const rawContent = await fs.readFile(filepath, 'utf-8').catch(() => null);
    const context = rawContent?.replace(/\r\n/g, '') || 'ไม่มีบริบท';

    let session = await this.prisma.chatSession.upsert({ 
      where: {summaryId: dto.summaryId}, 
      update: {}, 
      create: {summaryId: dto.summaryId}
    })

    const history = await this.prisma.chatMessage.findMany({ 
      where: { sessionId: session.id }, 
      orderBy: { createdAt: 'asc'}
    })

    const messages = [
      { role: 'system', content: `คุณเป็น AI assistant ที่ช่วยตอบคำถามเกี่ยวกับสรุปวิดีโอและสนทนากับผู้ใช้ ตอบเป็นภาษาไทย กระชับและตรงประเด็น

กฎสำคัญมาก:
1. จดจำทุกสิ่งที่ผู้ใช้บอกในประวัติการสนทนา เช่น ความชอบ ชื่อ ข้อมูลส่วนตัว
2. เมื่อผู้ใช้ถามว่าข้อมูลส่วนตัวของฉัน ให้ค้นหาในประวัติการสนทนาที่ role=user ว่าผู้ใช้เคยบอกไว้อย่างไร แล้วตอบตามนั้น

บริบทวิดีโอ:\n${context}` },
      ...history.map(m => ({role : m.role.toLowerCase(), content: m.content})),
      { role: 'user', content: dto.message}
    ]

    const subject = new Subject<MessageEvent>();
    let fullReply = '';

    // Call Ollama with streaming
    axios.post(this.ollamaApiUrl, {
      model: this.ollamaModel, 
      messages, 
      stream: true
    }, {
      responseType: 'stream'
    }).then(response => {
      response.data.on('data', (chunk: Buffer) => {
        try {
          const lines = chunk.toString().split('\n').filter(line => line.trim());
          for (const line of lines) {
            const json = JSON.parse(line);
            if (json.message?.content) {
              fullReply += json.message.content;
              subject.next({ data: { content: json.message.content, done: false } } as MessageEvent);
            }
            if (json.done) {
              // Stream finished - save to database
              this.saveMessages(session.id, dto.message, fullReply);
              subject.next({ data: { content: '', done: true, sessionId: session.id } } as MessageEvent);
              subject.complete();
            }
          }
        } catch (e) {
          // Ignore parse errors for incomplete chunks
        }
      });

      response.data.on('error', (err: Error) => {
        subject.error(err);
      });
    }).catch(err => {
      subject.error(err);
    });

    return subject.asObservable();
  }

  private async saveMessages(sessionId: number, userMessage: string, assistantReply: string) {
    await this.prisma.$transaction([
      this.prisma.chatMessage.createMany({
        data: [
          {role: 'USER', content: userMessage, sessionId},
          {role: 'ASSISTANT', content: assistantReply, sessionId}
        ]
      }),
      this.prisma.chatSession.update({
        where: {id: sessionId},
        data: {}
      })
    ]);
  }

  async create(dto: CreateChatDto) {
    // หาสรุปเอา transcript ไปใช้เป็น context
    const summary = await this.prisma.summary.findUnique({ where: {id: dto.summaryId}})
    if(!summary) {
      throw new Error('Summary not found');
    }

    const normalizedPath = summary.transcriptPath?.replace(/\\/g, '/');

    if (!normalizedPath) {
      return { status: 'summary_file_not_found' };
    }
    
    const filepath = path.resolve(normalizedPath || '');

    const rawContent = await fs.readFile(filepath, 'utf-8').catch(() => null);

    const context = rawContent?.replace(/\r\n/g, '') || 'ไม่มีบริบท';

    // ถ้าไม่มี sessionId → สร้าง ChatSession ใหม่
    let session = await this.prisma.chatSession.upsert({ where: {summaryId: dto.summaryId}, update: {}, create: {summaryId: dto.summaryId}})

    // ดึงประวัติข้อความทั้งหมดจาก session
    const history = await this.prisma.chatMessage.findMany({ where: {
      sessionId: session.id
    }, orderBy: { createdAt: 'asc'}})

    // craft messages
    const messages = [
      { role: 'system', content: `คุณเป็น AI assistant ที่ช่วยตอบคำถามเกี่ยวกับสรุปวิดีโอและสนทนากับผู้ใช้ ตอบเป็นภาษาไทย กระชับและตรงประเด็น

กฎสำคัญมาก:
1. จดจำทุกสิ่งที่ผู้ใช้บอกในประวัติการสนทนา เช่น ความชอบ ชื่อ ข้อมูลส่วนตัว
2. เมื่อผู้ใช้ถามว่าข้อมูลส่วนตัวของฉัน ให้ค้นหาในประวัติการสนทนาที่ role=user ว่าผู้ใช้เคยบอกไว้อย่างไร แล้วตอบตามนั้น

บริบทวิดีโอ:\n${context}` },
      ...history.map(m => ({role : m.role.toLowerCase(), content: m.content})),
      { role: 'user', content: dto.message}
    ]

    let res, reply;

    // call ollama
    try {
      res = await axios.post(this.ollamaApiUrl, {model: this.ollamaModel, messages, stream: false});
      reply = res.data.message?.content || '';
    } catch (error) {
      console.error('Error calling Ollama:', error);
      throw new Error('Failed to get response from AI');
    }

    // save latest question and reply
    await this.prisma.$transaction([
      this.prisma.chatMessage.createMany({
      data: [
        {role: 'USER', content: dto.message, sessionId: session.id},
        {role: 'ASSISTANT', content: reply, sessionId: session.id}
      ]
    }),
    this.prisma.chatSession.update({
      where: {id: session.id},
      data: {}
    })
    ]);

    return {
      reply, sessionId: session.id, timestamp: new Date().toISOString()
    }
  }

  async history(summaryId: string) {
    const session = await this.prisma.chatSession.findUnique({ where: { summaryId }, include: { messages: { orderBy: { createdAt: 'asc' }}}});
    return session?.messages || [];
  }
}
