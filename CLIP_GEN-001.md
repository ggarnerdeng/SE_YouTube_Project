# 📄 Document: YouTube Shorts Clip Generation Workflow  
**Document ID:** CLIPGEN-001  
**Version:** 1.0  
**Last Updated:** 2025-06-15  
**Maintainer:** Garner Deng  
**Project:** Engineering-Driven Content Creation Pipeline  

---

## 🔖 Overview  
This document outlines the standardized workflow for generating YouTube Shorts using AI-assisted tools (OpusClip, ChatGPT) and manual review steps. The goal is to automate and optimize video slicing, metadata enhancement, and publishing with repeatability, scalability, and algorithmic virality in mind.

---

## 📌 CLIPGEN-001.1 – Objectives

- Create high-performing YouTube Shorts through a structured, repeatable pipeline
- Apply engineering practices (modularization, automation, metrics) to digital content production
- Minimize human overhead while maximizing quality and discoverability
- Ensure each output is trackable, improvable, and legally compliant

---

## 🎬 CLIPGEN-001.2 – Workflow Steps

### **CLIPGEN-001.2.1 – Source Video Identification**
- **Input:** Long-form YouTube video (URL)
- **Action:** Evaluate content for short-form potential
- **Output:** Documented video URL and notes

---

### **CLIPGEN-001.2.2 – Timestamp Selection**
- **Input:** Target video
- **Action:** Manually identify compelling segments (30–60s)
- **Output:** Start and end timestamps with context labels

> Example:
> ```
> URL: https://youtube.com/xyz
> Clip #1: 02:31 – 03:12 (Title: "Why AI Fails at Humor")
> ```

---

### **CLIPGEN-001.2.3 – Clip Generation via OpusClip**
- **Tool:** [OpusClip](https://www.opus.pro/)
- **Input:** YouTube URL + timestamps
- **Action:** Submit request for Shorts-format video generation
- **Output:**
  - Short-form video file(s)
  - Auto-generated title
  - Transcript
  - Suggested description

---

### **CLIPGEN-001.2.4 – Metadata Enhancement via ChatGPT**
- **Tool:** ChatGPT Plus (GPT-4o)
- **Input:**
  - Title
  - Description
  - Transcript
- **Prompt Template:** `PROMPT-META-001`
- **Action:** Enhance title + description for engagement and SEO
- **Output:**
  - Improved title (CTR optimized)
  - Description with relevant hashtags (#shorts, etc.)
  - Optional: Add a call-to-action (CTA) or tease

---

### **CLIPGEN-001.2.5 – Final Metadata Integration**
- **Input:** Enhanced title + description
- **Action:** Prepare content for upload
- **Variants:**
  - **Desktop:** Standard upload with new metadata (no thumbnail selection)
  - **Mobile:** Use YouTube mobile app for manual thumbnail frame selection
- **Output:** Finalized Shorts video ready for publishing

---

### **CLIPGEN-001.2.6 – Thumbnail Optimization (Mobile Only)**
- **Action:** Use scrubber to manually select the thumbnail frame
- **Best Practices:**
  - Prioritize clear expressions, motion, text overlays
  - Avoid blurry, dull, or low-contrast frames

> 🔁 *Thumbnails are only selectable via mobile upload at present — a critical step for maximizing Shorts CTR.*

---

## 🧩 CLIPGEN-001.3 – Optional Enhancements (Planned)

- `CLIPGEN-002` – Automate transcript + metadata submission to ChatGPT via API
- `CLIPGEN-003` – Integrate MySQL/Google Sheets to log:
  - Timestamps used
  - Metadata performance
  - Hashtag effectiveness
- `CLIPGEN-004` – Build thumbnail effectiveness dataset for future model training

---

## 📦 CLIPGEN-001.4 – File Structure (Proposed)

