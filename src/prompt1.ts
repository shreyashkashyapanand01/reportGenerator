import { LRUCache } from 'lru-cache';

// Simplified cache config (v11 compatible)
const promptCache = new LRUCache<string, string>({
  max: 100,
  ttl: 60 * 60 * 1000, // 1 hour TTL
  allowStale: false,
  updateAgeOnGet: true
});

export const systemPrompt = () => {
  const now = new Date();
  const hour = now.getHours();

  const researchPhase = () => {
    return hour < 6 ? 'Night: Batch processing' :
             hour < 12 ? 'Morning: Discovery' :
             hour < 18 ? 'Afternoon: Validation' :
             'Evening: Synthesis';
  };

  const instructions = [
    "PRISMA systematic reviews w/ source hierarchy: peer-reviewed > preprints > reports > media",
    "CASP quality checks + thematic/meta-analysis",
    "Contradiction documentation w/ evidence weighting",
    "Authoritative source prioritization + technical depth",
    "Doctorate-level analysis w/ innovation forecasting",
    "Structured explanations + citation rigor",
    "Glossary appendix + emerging tech exploration",
    "Speculative content labeling + iterative integration",
    "Markdown formatting + self-correcting workflow"
  ];

  // UPDATED: Added Key Learnings and References to the formal structure
  const responseFormat = [
    "**Response Structure:**",
    "1. Full Markdown Report (Single Output)",
    "2. All sections from the TOC must be present and ordered correctly.",
    "3. Key Learnings must be synthesized from findings and placed at the end.",
    "4. References must be a numbered list of full source URIs and Titles."
  ].join("\n");

  // Add time-based research parameters
  const timeContext = hour < 12
    ? "Morning Research Protocol: Aggressive discovery phase"
    : "Afternoon Research Protocol: Validation & synthesis focus";

  return `## Research Agent Configuration
**Temporal Context:** ${timeContext}
**System Time:** ${now.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short'
  })}
**Research Protocol:** v2.3.2 (STRUCTURAL FIX)
**Model Capabilities:** ${process.env.GEMINI_MODEL || "gemini-2.5-flash"}
**Research Phase:** ${researchPhase()}
**Cognitive Load Profile:** ${hour < 12 ? 'High creativity' : 'High accuracy'} mode

${responseFormat}

## Operational Parameters
${instructions.map((i, idx) => `${idx + 1}. ${i}`).join("\n")}`;
};

export const serpQueryPromptTemplate = `## Search Strategy Configuration
**Query Type Distribution:**
- Informational: 40%
- Comparative: 25%
- Technical: 20%
- Exploratory: 15%

**Response Format Requirements:** (Return ONLY JSON matching this schema. No extra text.)
\`\`\`json
{
  "queries": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "maxLength": 60},
        "type": {"type": "string", "enum": ["informational", "comparative", "technical", "exploratory"]},
        "expectedSources": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["query", "type"]
    },
    "minItems": {{numQueries}},
    "maxItems": {{numQueries}}
  }
}
\`\`\``;

export const learningPromptTemplate = `## Content Analysis Protocol
**Schema Version:** 2.1.0
**Error Handling:**
- 5001: Invalid source credibility score
- 5002: Missing fact verification
- 5003: Technical term mismatch

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "errorCodes": {
    "5001": "Source score must be 1-5",
    "5002": "Minimum 3 verifications required",
    "5003": "Undefined technical terms present"
  },
  "type": "object",
  "properties": {
    "learnings": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 280,
        "pattern": "^[A-Z][^.]{10,255}\\."
      }
    }
  },
  "required": ["learnings"],
  "additionalProperties": false
}

Return ONLY JSON that matches this schema. No extra text.`;

export const feedbackPromptTemplate = `## Query Refinement Matrix
**Quality Metrics:**
- Specificity Score (1-5)
- Research Depth Potential
- Answerability Estimate

**Response Schema:** (Return ONLY JSON matching this schema. No extra text.)
\`\`\`json
{
  "followUpQuestions": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "question": {"type": "string"},
        "purpose": {"type": "string", "enum": ["scope", "context", "precision"]},
        "criticality": {"type": "number", "minimum": 1, "maximum": 3}
      }
    }
  }
}
\`\`\``;

// --- CRITICAL FIX: REVISED generateGeminiPrompt FUNCTION ---
export const generateGeminiPrompt = ({
  query,
  researchGoal,
  learnings
}: {
  query: string,
  researchGoal: string,
  learnings: string[]
}): string => {
  const learningSummary = learnings.length > 0
    ? `Initial Learnings: ${learnings.join('; ')}`
    : 'No initial learnings provided.';

  return `
**TASK:** Generate a comprehensive, doctorate-level research report based on the following Research Goal, Query, and provided search results.

**PRIMARY REQUIREMENT:** The ENTIRE output must be a single Markdown block that STRICTLY follows the "Final Report Structure" below. Do not deviate from the headings and numbering.

**INPUTS:**
- **Research Goal:** ${researchGoal}
- **Initial Query:** ${query}
- **Learnings/Synthesis Input:** ${learningSummary}
- **Search Results (Provided by Tool):** [The final prompt will contain the concatenated search results here]

---
## Final Report Structure (STRICTLY adhere to this template)
---

# [Report Title: Synthesize a professional title from the Research Goal]

## Abstract
[Synthesize a concise, 1-2 paragraph summary of the entire report's objective, methodology, and key findings.]

## Table of Contents
[Generate a numbered Table of Contents for sections I through VII.]

# I. Introduction
[Introduce the topic, state the research objective, and define the scope/key terms.]

# II. Methodology Description
[**MANDATORY:** Detail the data collection strategy, screening criteria, success metric application, and analytical framework used to conduct the research.]

# III. Executive Summary
[Synthesize 3-5 high-impact bullet points summarizing the most critical findings and market trends for an executive audience.]

# IV. Key Findings: Analysis and Data Presentation
[Present the core analysis, detailed metrics, and supporting evidence from the research. Use clear sub-headers (A, B, C...) as needed.]

# V. Limitations Disclosure
[**MANDATORY:** Clearly articulate the constraints of the research, including data availability, subjective definitions, and external market factors.]

# VI. Future Research Directions
[Suggest 3-5 logical next steps or extensions for this line of inquiry.]

# VII. Glossary of Technical Terms
[Define all specialized terms used in the report for clarity.]

---
## Final Synthesis and Source Accountability
---

## Key Learnings
[**MANDATORY:** Synthesize the 3-5 most important, high-level takeaways (1-2 sentences each) from the entire report. This is the ultimate "So what?" section.]

## References
[**MANDATORY:** Use the source data provided by the search tool to generate a numbered list of all **full URLs and Titles** used to ground the report. Do not use generic placeholders.]
`;
};
// --- END REVISED generateGeminiPrompt FUNCTION ---

// Keep only what's actually used
export const clearPromptCache = () => promptCache.clear();


// Add validation layer
const validatePromptConsistency = () => {
  const templates = [systemPrompt, serpQueryPromptTemplate, learningPromptTemplate];
  const versions = templates.map(t => {
    const content = typeof t === 'function' ? t() : t;
    // Update regex to reflect the new version format in systemPrompt
    const match = content.match(/Protocol: v(\d+\.\d+\.\d+)/)?.[1] || content.match(/Schema Version: (\d+\.\d+\.\d+)/)?.[1];
    return match;
  });

  // Filter out any undefined matches to avoid false positives if a template is incomplete
  const definedVersions = versions.filter(v => v);

  if (new Set(definedVersions).size > 1) {
    // Log error but don't crash on prompt version mismatch for robustness
    console.error(`Prompt version mismatch: ${definedVersions.join(' vs ')}`);
  }
};
validatePromptConsistency(); // Run at module load
