-- CreateEnum
CREATE TYPE "SummaryStatus" AS ENUM ('QUEUED', 'RUNNING', 'DONE', 'ERROR', 'CANCEL');

-- CreateTable
CREATE TABLE "User" (
    "id" SERIAL NOT NULL,
    "username" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Summary" (
    "id" TEXT NOT NULL,
    "youtubeUrl" TEXT NOT NULL,
    "status" "SummaryStatus" NOT NULL DEFAULT 'QUEUED',
    "percent" INTEGER NOT NULL DEFAULT 0,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),
    "transcriptPath" TEXT,
    "bulletPath" TEXT,
    "summaryPath" TEXT,
    "sceneFactsPath" TEXT,
    "whisperModel" TEXT,
    "asrDevice" TEXT,
    "vlDevice" TEXT,
    "vlModel" TEXT,
    "sceneThresh" DOUBLE PRECISION,
    "enableOcr" BOOLEAN,
    "frames" INTEGER,
    "bulletCount" INTEGER,
    "transcriptWord" INTEGER,
    "summaryWord" INTEGER,
    "timeDownloadSec" DOUBLE PRECISION,
    "timeSpeechtoTextSec" DOUBLE PRECISION,
    "timeCaptionImageSec" DOUBLE PRECISION,
    "timeSummarizeSec" DOUBLE PRECISION,
    "timeTotal" DOUBLE PRECISION,
    "durationSec" DOUBLE PRECISION,
    "errorMessage" TEXT,
    "keyword" TEXT,
    "userId" INTEGER NOT NULL,

    CONSTRAINT "Summary_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OntologyTopic" (
    "id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "frequency" INTEGER NOT NULL DEFAULT 0,
    "score" DOUBLE PRECISION,
    "relatedTopics" TEXT[] DEFAULT ARRAY[]::TEXT[],

    CONSTRAINT "OntologyTopic_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserOntologyTopic" (
    "userId" INTEGER NOT NULL,
    "topicId" UUID NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserOntologyTopic_pkey" PRIMARY KEY ("userId","topicId")
);

-- CreateTable
CREATE TABLE "OntologyTopicRelation" (
    "fromTopicId" UUID NOT NULL,
    "toTopicId" UUID NOT NULL,
    "weight" DOUBLE PRECISION,

    CONSTRAINT "OntologyTopicRelation_pkey" PRIMARY KEY ("fromTopicId","toTopicId")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_username_key" ON "User"("username");

-- CreateIndex
CREATE INDEX "Summary_status_startedAt_idx" ON "Summary"("status", "startedAt");

-- CreateIndex
CREATE UNIQUE INDEX "OntologyTopic_name_key" ON "OntologyTopic"("name");

-- CreateIndex
CREATE INDEX "UserOntologyTopic_userId_idx" ON "UserOntologyTopic"("userId");

-- CreateIndex
CREATE INDEX "UserOntologyTopic_topicId_idx" ON "UserOntologyTopic"("topicId");

-- AddForeignKey
ALTER TABLE "Summary" ADD CONSTRAINT "Summary_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserOntologyTopic" ADD CONSTRAINT "UserOntologyTopic_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserOntologyTopic" ADD CONSTRAINT "UserOntologyTopic_topicId_fkey" FOREIGN KEY ("topicId") REFERENCES "OntologyTopic"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OntologyTopicRelation" ADD CONSTRAINT "OntologyTopicRelation_fromTopicId_fkey" FOREIGN KEY ("fromTopicId") REFERENCES "OntologyTopic"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OntologyTopicRelation" ADD CONSTRAINT "OntologyTopicRelation_toTopicId_fkey" FOREIGN KEY ("toTopicId") REFERENCES "OntologyTopic"("id") ON DELETE CASCADE ON UPDATE CASCADE;
