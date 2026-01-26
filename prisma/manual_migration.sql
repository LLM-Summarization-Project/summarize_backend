-- สร้างตาราง SummaryOwner
CREATE TABLE IF NOT EXISTS "SummaryOwner" (
    "summaryId" TEXT NOT NULL,
    "userId" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "SummaryOwner_pkey" PRIMARY KEY ("summaryId", "userId")
);

-- สร้าง foreign keys
ALTER TABLE "SummaryOwner" ADD CONSTRAINT "SummaryOwner_summaryId_fkey" 
    FOREIGN KEY ("summaryId") REFERENCES "Summary"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "SummaryOwner" ADD CONSTRAINT "SummaryOwner_userId_fkey" 
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- สร้าง index
CREATE INDEX IF NOT EXISTS "SummaryOwner_userId_idx" ON "SummaryOwner"("userId");

-- ย้ายข้อมูลจาก userId เดิม
INSERT INTO "SummaryOwner" ("summaryId", "userId", "createdAt")
SELECT "id", "userId", "startedAt"
FROM "Summary"
WHERE "userId" IS NOT NULL
ON CONFLICT DO NOTHING;