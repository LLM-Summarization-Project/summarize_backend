import { Injectable } from '@nestjs/common';
import { CreateChatDto } from './dto/create-chat.dto';
import axios from 'axios';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from 'src/summarize/prisma/prisma.service';
import path from 'path';
import * as fs from 'fs/promises';

@Injectable()
export class ChatService {
  private ollamaApiUrl: string;
  private ollamaModel: string;
  
  constructor(configService: ConfigService, private prisma: PrismaService) {
    this.ollamaApiUrl = configService.get<string>('OLLAMA_API_CHAT') || 'http://localhost:11434/api/chat';
    this.ollamaModel = configService.get<string>('OLLAMA_MODEL') || 'scb10x/llama3.1-typhoon2-8b-instruct';
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
      { role: 'system', content: `คุณเป็น AI assistant ที่ช่วยตอบคำถามเกี่ยวกับสรุปวิดีโอ ตอบเป็นภาษาไทย กระชับและตรงประเด็น บริบท:\n${context}` },
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
