import { LRUCache } from 'lru-cache';
import { z } from 'zod';

import { callGeminiProConfigurable } from './ai/providers.js';
import { OutputManager } from './output-manager.js';
import { feedbackPromptTemplate, systemPrompt } from './prompt.js';
import {
  validateAcademicInput,
  validateAcademicOutput,
  conductResearch
} from './deep-research.js';
import {
  generateProgressBar as terminalProgressBar,
  TERMINAL_CONTROLS,
  getTerminalDimensions
} from './terminal-utils.js';

interface FeedbackOptions {
  query: string;
  numQuestions?: number;
  researchGoal?: string;
  depth?: number;
  breadth?: number;
  existingLearnings?: string[];
}

const output = new OutputManager();

// Optimized cache configuration
const FEEDBACK_CACHE_CONFIG = {
  max: 100,
  ttl: 300_000, // 5 minutes
  dispose: (value: FeedbackResponse) => {
    OutputManager.logCacheEviction(value);
  }
};

const feedbackCache = new LRUCache<string, FeedbackResponse>(FEEDBACK_CACHE_CONFIG);

// Enhanced validation schema
const FeedbackResponseSchema = z.object({
  followUpQuestions: z.array(z.string().min(10)).max(5),
  analysis: z.string().min(100),
  confidenceScore: z.number().min(0).max(1).optional()
}).passthrough();

export type FeedbackResponse = z.infer<typeof FeedbackResponseSchema>;

interface FeedbackAnalysis {
  readonly [key: string]: unknown;
  readonly score: number;
  readonly keywords: readonly string[];
  readonly suggestions: string[];
  readonly analysisDate: Date;
}

const DEFAULT_KEYWORDS = ['example', 'test', 'dummy'] as const;

async function analyzePastFeedback(query: string): Promise<FeedbackAnalysis> {
  const { width } = getTerminalDimensions();
  const analysisProgress = terminalProgressBar({
    current: 0,
    total: 100,
    width: width - 20,
    label: 'Analyzing feedback'
  });

  try {
    process.stdout.write(`${TERMINAL_CONTROLS.cursorHide}${analysisProgress}`);
    
    // Validate input type and content
    if (typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Invalid query: Must be a non-empty string');
    }

    const startTime = performance.now();
    const matchedKeywords: string[] = [];
    const suggestions: string[] = [];
    
    // Use find() for early exit instead of for-loop
    const foundKeyword = DEFAULT_KEYWORDS.find(keyword => 
      query.toLowerCase().includes(keyword.toLowerCase())
    );

    if (foundKeyword) {
      matchedKeywords.push(foundKeyword);
      suggestions.push(`Consider refining the query to be more specific than "${foundKeyword}"`);
    }

    const analysis: FeedbackAnalysis = {
      score: calculateFeedbackScore(matchedKeywords),
      keywords: [...matchedKeywords] as readonly string[],
      suggestions,
      analysisDate: new Date(),
    };

    // Performance metrics
    const analysisTime = performance.now() - startTime;
    
    output.log(`Analyzed feedback for "${query}" in ${analysisTime.toFixed(2)}ms`, {
      score: analysis.score,
      keywords: analysis.keywords
    });

    return analysis;
  } finally {
    process.stdout.write(TERMINAL_CONTROLS.cursorShow);
  }
}

// Utility function with explicit return type
function calculateFeedbackScore(keywords: readonly string[]): number {
  const baseScore = 100;
  const penalty = keywords.length * 10;
  return Math.max(baseScore - penalty, 0);
}

// Unified progress bar generator
type ProgressBarParams = {
  current: number;
  total: number;
  width?: number;
  label: string;
};

function generateProgressBar({
  current,
  total,
  width = 30,
  label
}: ProgressBarParams): string {
  const filled = Math.round((width * current) / total);
  return `${label} [${'█'.repeat(filled)}${' '.repeat(width - filled)}]`;
}

export async function generateFeedback({
  query,
  numQuestions = 3,
  researchGoal = "Understand the user's query",
  depth = 1,
  breadth = 1,
  existingLearnings = [],
}: FeedbackOptions): Promise<FeedbackResponse> {
  // Validate input before processing
  if (!validateAcademicInput(query)) {
    throw new Error('Query fails academic validation checks');
  }

  // 1. Create cache key
  let cacheKey: string;
  try {
    const keyObject: Record<string, unknown> = { query, numQuestions, researchGoal, depth, breadth, existingLearnings };
    // Omit default values for a more efficient key (optional, as before)
    if (numQuestions === 3) {
      delete keyObject.numQuestions;
    }
    if (researchGoal === "Understand the user's query") {
      delete keyObject.researchGoal;
    }
    if (depth === 1) {
      delete keyObject.depth;
    }
    if (breadth === 1) {
      delete keyObject.breadth;
    }
    const learningsHash = existingLearnings.length > 0 ? String(existingLearnings.reduce((acc, val) => acc + val.charCodeAt(0), 0)) : ''; // Hash learnings
    keyObject.learningsHash = learningsHash;
    cacheKey = JSON.stringify(keyObject);
  } catch (keyError) {
    output.log("Error creating feedback cache key:", { error: keyError instanceof Error ? keyError.message : String(keyError) });
    cacheKey = 'default-feedback-key'; // Fallback key in case of error
  }

  // 2. Check cache
  try {
    const cachedFeedback = feedbackCache.get(cacheKey);
    if (cachedFeedback) {
      output.log("Cache hit:", { key: cacheKey });
      return cachedFeedback;
    }
  } catch (cacheGetError) {
    output.log("Cache error:", {
      error: cacheGetError instanceof Error ? cacheGetError.message : 'Unknown cache error'
    });
  }

  const context = `
Research Goal: ${researchGoal}
Current Depth: ${depth}
Current Breadth: ${breadth}
Existing Learnings: ${existingLearnings.join('\n')}
`;

  // Get feedback from past interactions
const researchResult = await conductResearch(query, depth);
const analysisText = researchResult?.analysis?.trim() || query;
const pastFeedback = await analyzePastFeedback(analysisText);

  // Use feedbackPromptTemplate and replace variables correctly
  const geminiPrompt = `${systemPrompt()}\n\n${context}\n\n${pastFeedback}\n\n${feedbackPromptTemplate
    .replace("{{query}}", query) // Use "{{query}}" in feedbackPromptTemplate, not "{{userQuery}}"
    .replace("{{numQuestions}}", String(numQuestions)) // Replace numQuestions placeholder
  }`;

  let feedbackResponse: FeedbackResponse = {
    analysis: "Initial feedback response.",
    followUpQuestions: []
  };
  const { width } = getTerminalDimensions();
  let localProgress = 0;
  const totalSteps = 3; // prompt build, API+parse, cache
  try {
    output.log(`Generating feedback for query: "${query}"...`); // Log using OutputManager
    // Render initial progress bar (uses local generateProgressBar)
    try {
      const bar = generateProgressBar({ current: localProgress, total: totalSteps, width: Math.max(10, width - 20), label: 'Feedback' });
      process.stdout.write(`${TERMINAL_CONTROLS.cursorHide}${bar}`);
    } catch {}

    const { FeedbackResponseJsonSchema } = await import('./types.js');
    localProgress = 1;
    try {
      const bar = generateProgressBar({ current: localProgress, total: totalSteps, width: Math.max(10, width - 20), label: 'Feedback' });
      process.stdout.write(`\r${bar}`);
    } catch {}
    let jsonText = await callGeminiProConfigurable(
      `${systemPrompt()}\n\nReturn ONLY JSON matching the schema.\n${geminiPrompt}`,
      { json: true, schema: FeedbackResponseJsonSchema, tools: [{ googleSearch: {} }] }
    );

    // First parse attempt
    let parsed: unknown;
    try {
      parsed = JSON.parse(jsonText);
      feedbackResponse = FeedbackResponseSchema.parse(parsed);
      output.log(`Parsed Feedback Response (JSON):\n${JSON.stringify(feedbackResponse, null, 2)}`);
    } catch (jsonError) {
      output.log(`Feedback JSON parse/validation error: ${jsonError}`);
      output.log(`Attempting single repair pass.`);
      // One repair pass: ask model to fix to schema
      jsonText = await callGeminiProConfigurable(
        `${systemPrompt()}\n\nYour previous output did not match the schema. Repair it to valid JSON that matches strictly. Return ONLY JSON.\nSchema: ${JSON.stringify(FeedbackResponseJsonSchema)}\nPrevious:\n${jsonText}`,
        { json: true, schema: FeedbackResponseJsonSchema, tools: [{ googleSearch: {} }] }
      );
      try {
        parsed = JSON.parse(jsonText);
        feedbackResponse = FeedbackResponseSchema.parse(parsed);
        output.log(`Repair pass succeeded.`);
      } catch (repairError) {
        output.log(`Repair pass failed: ${repairError}`);
        feedbackResponse = {
          analysis: "Failed to parse feedback response after repair.",
          followUpQuestions: []
        };
      }
    }

    // Update progress for cache step
    localProgress = 2;
    try {
      const bar = generateProgressBar({ current: localProgress, total: totalSteps, width: Math.max(10, width - 20), label: 'Feedback' });
      process.stdout.write(`\r${bar}`);
    } catch {}

    // Validate output before returning
    const isValidOutput = validateAcademicOutput(feedbackResponse.analysis || '');
    if (!isValidOutput) {
      output.log('Generated feedback failed academic validation');
      return {
        analysis: 'Unable to generate valid feedback',
        followUpQuestions: [] // Required by Zod schema
      };
    }

    // 3. Store in cache after successful API call and parsing
    try {
      feedbackCache.set(cacheKey, feedbackResponse);
      output.log(`Cached feedback for key: ${cacheKey}`);
    } catch (cacheSetError: unknown) {
      output.log("Error setting feedback to cache:", { error: cacheSetError instanceof Error ? cacheSetError.message : String(cacheSetError) });
    }
  } catch (apiError: unknown) {
    output.log(`Gemini API error during feedback generation: ${apiError}`);
    feedbackResponse = {
      analysis: "Error generating feedback. Please check API key and logs.",
      followUpQuestions: [] // Add required array
    }; // API error fallback
  } finally {
    try {
      // Complete bar to 100%
      const bar = generateProgressBar({ current: totalSteps, total: totalSteps, width: Math.max(10, width - 20), label: 'Feedback' });
      process.stdout.write(`\r${bar}${TERMINAL_CONTROLS.cursorShow}\n`);
    } catch {
      process.stdout.write(TERMINAL_CONTROLS.cursorShow);
    }
  }
  return feedbackResponse; // Return the feedback response object (either parsed or fallback)
}