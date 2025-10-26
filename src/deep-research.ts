import { compact, escape as lodashEscape } from 'lodash-es';
import { LRUCache } from 'lru-cache';
import pLimit from 'p-limit'; // Concurrency limiter
import { z } from 'zod';

import { callGeminiProConfigurable, generateBatch, generateBatchWithTools, trimPrompt } from './ai/providers.js';
import { RecursiveCharacterTextSplitter, SemanticTextSplitter } from './ai/text-splitter.js';
import { OutputManager } from './output-manager.js';
import { generateGeminiPrompt, learningPromptTemplate, serpQueryPromptTemplate, systemPrompt } from './prompt.js';
import { extractJsonFromText, isValidJSON, safeParseJSON, stringifyJSON } from './utils/json.js';
import { sanitizeReportContent } from './utils/sanitize.js';
import fs from 'fs';
const output = new OutputManager();


// Rename your local type to avoid conflict
export interface ResearchResult {
  content: string;
  sources: string[];
  methodology: string;
  limitations: string;
  citations: { reference: string; context?: string }[];
  learnings: string[];
  visitedUrls: string[];
  firecrawlResults: SearchResponse; // Use actual Firecrawl type
  analysis: string;
}

export interface ProcessResult {
  analysis: string;
  content: string;
  sources: string[];
  methodology: string;
  limitations: string;
  citations: string[];
  learnings: string[];
  visitedUrls: string[];
  firecrawlResults: SearchResponse;
}

export interface ResearchProgress {
  [key: string]: unknown; // Add index signature
  currentQuery?: string;
  currentDepth: number;
  totalDepth: number;
  currentBreadth: number;
  totalBreadth: number;
  totalQueries: number;
  completedQueries: number;
  progressMsg?: string;
}

export interface researchProgress {
  progressMsg: string;
}

// Configuration from environment variables
const {GEMINI_API_KEY} = process.env;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash"; // Default to gemini-pro

const CONCURRENCY_LIMIT = parseInt(process.env.CONCURRENCY_LIMIT || "5", 10); // Default to 5

if (!GEMINI_API_KEY) {


  throw new Error('GEMINI_API_KEY environment variable is not set');
}


const ConcurrencyLimit = CONCURRENCY_LIMIT;

// take en user query, return a list of SERP queries
const SerpQuerySchema = z.object({
  query: z.string(),
  researchGoal: z.string(),

});

type SerpQuery = { query: string; researchGoal: string; };

const DEFAULT_NUM_QUERIES = 3;

// Create an LRU cache instance
const serpQueryCache = new LRUCache<string, SerpQuery[]>({
  max: 50, // Maximum number of items in the cache
});

// Create LRU cache for final reports
const reportCache = new LRUCache<string, { __returned: string; __abortController: AbortController; __staleWhileFetching: undefined }>({ // Cache stores report strings

  max: 20, // Adjust max size as needed
});

function logResearchProgress(progressData: ResearchProgress) {
  try {
    const prettyJson = JSON.stringify(progressData, null, 2);
    output.log("Progress:", JSON.parse(prettyJson)); // Parse before logging
  } catch (e) {
    output.log("Log error:", { error: e instanceof Error ? e.message: String(e) });
    output.log("Raw data:", { data: progressData });
  }
}

function cacheSearchResults(query: string, results: unknown) {
  const minifiedResultsJson = stringifyJSON(results); // Minified for efficient storage

  if (minifiedResultsJson) {
    // ... code to store minifiedResultsJson in your cache (e.g., LRUCache) ...
    output.log(`Cached results for query: "${query}" (JSON length: ${minifiedResultsJson.length})`);
  } else {
    output.log("Error stringifying search results for caching.");
    // Handle caching error
  }
}

async function generateSerpQueries({
  query: rawQuery,
  numQueries = DEFAULT_NUM_QUERIES,
  learnings = [],
  researchGoal = "Initial query",
  initialQuery,
  depth = 1,
  breadth = 1,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
  researchGoal?: string;
  initialQuery?: string;
  depth?: number;
  breadth?: number;
}): Promise<SerpQuery[]> {
  try {
    // 1. Create a cache key
    let cacheKey: string;
    try {
      const keyObject: Record<string, unknown> = { rawQuery, numQueries, researchGoal, initialQuery, depth, breadth };

      // Omit default values from the cache key
      if (numQueries === DEFAULT_NUM_QUERIES) {
        delete keyObject.numQueries;
      }
      if (researchGoal === "Initial query") {
        delete keyObject.researchGoal;
      }
      if (initialQuery === rawQuery) {
        delete keyObject.initialQuery;
      } // Assuming initialQuery defaults to rawQuery
      if (depth === 1) {
        delete keyObject.depth;
      }
      if (breadth === 1) {
        delete keyObject.breadth;
      }

      // Hash the learnings array (example using a simple hash function)
      const learningsHash = learnings.length > 0 ? String(learnings.reduce((acc, val) => acc + val.charCodeAt(0), 0)) : '';
      keyObject.learningsHash = learningsHash;

      cacheKey = JSON.stringify(keyObject);
    } catch (e) {
      output.log("Error creating cache key:", { error: e instanceof Error ? e.message : 'Unknown error' });
      cacheKey = rawQuery; // Fallback to a simple key
    }

    // 2. Check if the result is in the cache
    try {
      const cachedResult = serpQueryCache.get(cacheKey);
      if (cachedResult) {
        output.log("Cache hit:", { key: cacheKey });
        return cachedResult;
      }
    } catch (e) {
      output.log("Cache error:", { error: e instanceof Error ? e.message : 'Unknown cache error' });
    }

    output.log(`Generating SERP queries for key: ${cacheKey}`);
    const query = lodashEscape(rawQuery);
    const sanitizedLearnings = Array.isArray(learnings) ? learnings.map(lodashEscape) : [];
    const prompt = serpQueryPromptTemplate
      .replace('{{query}}', query)
      .replace('{{numQueries}}', String(numQueries))
      .replace('{{researchGoal}}', researchGoal || "General Research")
      .replace('{{initialQuery}}', initialQuery || rawQuery)
      .replace('{{depth}}', depth?.toString() || "1")
      .replace('{{breadth}}', breadth?.toString() || "1");
    let learningsString = '';
    if (Array.isArray(sanitizedLearnings) && sanitizedLearnings.length > 0) {
      learningsString = sanitizedLearnings.join('\n');
    }
    const finalPrompt = prompt.replace('{{learnings.join("\\n")}}', learningsString);
    output.log(`generateSerpQueries prompt: ${finalPrompt}`);
    let jsonString: string = '{}';
    try {
      const geminiText = await callGeminiProConfigurable(finalPrompt, { tools: [{ googleSearch: {} }] });
      if (typeof geminiText === 'string') {
        jsonString = geminiText;
      } else {
        output.log("Error: Gemini response text is not a string or is missing.");
      }
    } catch (err) {
      output.log("Gemini error:", { error: err instanceof Error ? err.message : 'Unknown error' });
      output.log('Error in generateSerpQueries:', { error: err instanceof Error ? err.message : 'Unknown error' });
      jsonString = '{}';
    }
    if (!isValidJSON(jsonString)) {
      output.log("Gemini returned non-JSON text; attempting JSON extraction.");
    }
    const rawQueriesJSON = extractJsonFromText(jsonString);
    let serpQueries: SerpQuery[] = [];

    if (rawQueriesJSON && Array.isArray(rawQueriesJSON)) {
      serpQueries = rawQueriesJSON
        .slice(0, numQueries)
        .map((rawQuery: unknown) => {
          const queryValue = ((): unknown => {
            if (typeof rawQuery === 'object' && rawQuery !== null && 'query' in rawQuery) {
              const q = (rawQuery as { query?: unknown }).query;
              return typeof q === 'string' ? q : q != null ? String(q) : '';
            }
            return rawQuery;
          })();
          const parsed = SerpQuerySchema.safeParse({
            query: typeof queryValue === 'string' ? queryValue : String(queryValue ?? ''),
            researchGoal,
          });
          return parsed.success ? (parsed.data as SerpQuery) : null;
        })
        .filter(Boolean) as SerpQuery[];
    } else {
      output.log("Failed to generate or parse SERP queries from Gemini response, using fallback to empty array.");
      serpQueries = [];
    }

    // 4. Store the result in the cache
    // Also persist a minified copy via cacheSearchResults for observability
    try {
      cacheSearchResults(rawQuery, serpQueries);
    } catch (e) {
      output.log("cacheSearchResults error:", { error: e instanceof Error ? e.message : 'Unknown error' });
    }
    try {
      serpQueryCache.set(cacheKey, serpQueries);
      output.log(`Cached SERP queries for key: ${cacheKey}`);
    } catch (e) {
      output.log("Error setting to cache:", { error: e instanceof Error ? e.message : 'Unknown error' });
    }

    return serpQueries;

  } catch (error) {
    output.log("Error in generateSerpQueries:", { error: error instanceof Error ? error.message : 'Unknown error' });
    return []; // Return an empty array in case of any error during query generation
  }
}



// async function generateSerpQueries({
//   query: rawQuery,
//   numQueries = DEFAULT_NUM_QUERIES,
//   learnings = [],
//   researchGoal = "Initial query",
//   initialQuery,
//   depth = 1,
//   breadth = 1,
// }: {
//   query: string;
//   numQueries?: number;
//   learnings?: string[];
//   researchGoal?: string;
//   initialQuery?: string;
//   depth?: number;
//   breadth?: number;
// }): Promise<SerpQuery[]> {
//   try {
//     // 1. Create cache key
//     let cacheKey: string;
//     try {
//       const keyObject: Record<string, unknown> = { rawQuery, numQueries, researchGoal, initialQuery, depth, breadth };
//       if (numQueries === DEFAULT_NUM_QUERIES) delete keyObject.numQueries;
//       if (researchGoal === "Initial query") delete keyObject.researchGoal;
//       if (initialQuery === rawQuery) delete keyObject.initialQuery;
//       if (depth === 1) delete keyObject.depth;
//       if (breadth === 1) delete keyObject.breadth;

//       const learningsHash = learnings.length > 0 ? String(learnings.reduce((acc, val) => acc + val.charCodeAt(0), 0)) : '';
//       keyObject.learningsHash = learningsHash;

//       cacheKey = JSON.stringify(keyObject);
//     } catch (e) {
//       output.log("Error creating cache key:", { error: e instanceof Error ? e.message : 'Unknown error' });
//       cacheKey = rawQuery;
//     }

//     // 2. Check cache
//     try {
//       const cached = serpQueryCache.get(cacheKey);
//       if (cached) {
//         output.log("Cache hit:", { key: cacheKey });
//         return cached;
//       }
//     } catch (e) {
//       output.log("Cache error:", { error: e instanceof Error ? e.message : 'Unknown cache error' });
//     }

//     // 3. Build prompt safely
//     const sanitizedLearnings = learnings.map(l => l.replace(/\n/g, ' ').trim());
//     const learningsString = sanitizedLearnings.join('\n');

//     const finalPrompt = `
// Generate ${numQueries} search queries for the following research goal:
// Query: "${rawQuery}"
// Research Goal: "${researchGoal}"
// Initial Query: "${initialQuery || rawQuery}"
// Depth: ${depth}, Breadth: ${breadth}
// Learnings so far:
// ${learningsString}
// Return output as a valid JSON array of objects: [{ "query": "query string", "researchGoal": "goal string" }]
// `;

//     output.log(`generateSerpQueries prompt:\n${finalPrompt.substring(0, 300)}...`);

//     // 4. Call Gemini
//     let jsonString = '{}';
//     try {
//       const geminiText = await callGeminiProConfigurable(finalPrompt, { tools: [{ googleSearch: {} }] });
//       jsonString = typeof geminiText === 'string' ? geminiText : '{}';
//     } catch (err) {
//       output.log("Gemini error:", { error: err instanceof Error ? err.message : 'Unknown error' });
//       jsonString = '{}';
//     }

//     // 5. Ensure valid JSON
//     if (!isValidJSON(jsonString)) {
//       output.log("Gemini returned non-JSON; extracting JSON portion.");
//       jsonString = extractJsonFromText(jsonString) || '[]';
//     }

//     let rawQueriesJSON: unknown[] = [];
//     try {
//       rawQueriesJSON = JSON.parse(jsonString);
//     } catch {
//       output.log("Failed to parse JSON, using empty array.");
//       rawQueriesJSON = [];
//     }

//     // 6. Parse and validate SERP queries
//     const serpQueries: SerpQuery[] = (rawQueriesJSON as unknown[])
//       .map(item => {
//         if (typeof item === 'object' && item !== null && 'query' in item) {
//           const parsed = SerpQuerySchema.safeParse({
//             query: String((item as { query?: unknown }).query ?? ''),
//             researchGoal,
//           });
//           return parsed.success ? parsed.data : null;
//         }
//         return null;
//       })
//       .filter(Boolean) as SerpQuery[];

//     // 7. Cache results
//     try {
//       serpQueryCache.set(cacheKey, serpQueries);
//       cacheSearchResults(rawQuery, serpQueries);
//       output.log(`Cached SERP queries for key: ${cacheKey}`);
//     } catch (e) {
//       output.log("Error caching SERP queries:", { error: e instanceof Error ? e.message : 'Unknown error' });
//     }

//     return serpQueries;
//   } catch (error) {
//     output.log("Error in generateSerpQueries:", { error: error instanceof Error ? error.message : 'Unknown error' });
//     return [];
//   }
// }


// Update splitter initialization to use provider configuration

const createResearchSplitter = () => {
  return new RecursiveCharacterTextSplitter({
    chunkSize: 140,
    chunkOverlap: 20,
    separators: ['\n\n', '\n', ' ']
  });

};



// Update processSerpResult to use provider-sanctioned splitter

//----------------------------------------------//

async function processSerpResult({
  query,
  result,
  numLearnings = 3,
}: {
  query: string;
  result: SearchResponse;
  numLearnings?: number;
}): Promise<{ learnings: string[]; followUpQuestions: string[]; }> {
  const contents = compact(result.data.map(item => item.markdown ?? ''));
  const resolvedContents = await Promise.all(contents);

  const urls = compact(result.data.map(item => item.url));
  output.log(`Ran ${query}, found ${contents.length} contents and ${urls.length} URLs:`, { urls });

  // Use provider-approved text splitter
  const splitter = createResearchSplitter();
  let chunks: string[] = [];

  // Add proper error handling for splitter failures
  try {
    chunks = await splitter.splitText(resolvedContents.join("\n\n"));
  } catch (error) {
    output.log(`Text splitting failed: ${error}`);
    return { learnings: [], followUpQuestions: [] };

  }



  // Process each chunk with the LLM (batched)
  const firstUrl = result?.data?.[0]?.url;
  let derivedTitle = 'No Title';
  if (typeof firstUrl === 'string') {
    try {
      derivedTitle = new URL(firstUrl).hostname || firstUrl;
    } catch {
      derivedTitle = firstUrl;
    }
  }

  const prompts = chunks.map((chunk) =>
    learningPromptTemplate
      .replace("{{query}}", query)
      .replace("{{title}}", derivedTitle)
      .replace("{{url}}", typeof firstUrl === 'string' ? firstUrl : "No URL")
      .replace("{{content}}", chunk)
  );
  const batchResults = await generateBatchWithTools(prompts, [{ googleSearch: {} }]);
  const learnings: string[] = [];
  for (const br of batchResults) {
    const text = await br.response.text();
    const parsedResult = z.object({
      learnings: z.array(z.string()),
      followUpQuestions: z.array(z.string()),
    }).safeParse(JSON.parse(text));
    if (parsedResult.success) {

      learnings.push(...(parsedResult.data.learnings ?? []));

    }
  }

  return { learnings: learnings.slice(0, numLearnings) ?? [], followUpQuestions: [] };
}

//------------------------------------------------------//


// async function processSerpResult({
//   query,
//   result,
//   numLearnings = 3,
// }: {
//   query: string;
//   result: SearchResponse;
//   numLearnings?: number;
// }): Promise<{ learnings: string[]; followUpQuestions: string[]; }> {

//   // --- START: ADDED LOGGING CODE ---
//   const logFilePath = 'processSerpResult_log.txt';
//   const timestamp = new Date().toISOString();

//   // Prepare the input data for logging
//   const inputLog = `
// =============================================
// TIMESTAMP: ${timestamp}

// --- INPUT ---
// Query: ${query}
// Result (URLs Scraped): ${JSON.stringify(result.data.map(item => item.url), null, 2)}
// Number of Learnings to Find: ${numLearnings}
// ---------------------------------------------
// `;
//   // --- END: ADDED LOGGING CODE ---

//   const contents = compact(result.data.map(item => item.markdown ?? ''));
//   const resolvedContents = await Promise.all(contents);
//   const urls = compact(result.data.map(item => item.url));
//   output.log(`Ran ${query}, found ${contents.length} contents and ${urls.length} URLs:`, { urls });

//   const splitter = createResearchSplitter();
//   let chunks: string[] = [];

//   try {
//     chunks = await splitter.splitText(resolvedContents.join("\n\n"));
//   } catch (error) {
//     output.log(`Text splitting failed: ${error}`);
//     return { learnings: [], followUpQuestions: [] };
//   }

//   const firstUrl = result?.data?.[0]?.url;
//   let derivedTitle = 'No Title';
//   if (typeof firstUrl === 'string') {
//     try {
//       derivedTitle = new URL(firstUrl).hostname || firstUrl;
//     } catch {
//       derivedTitle = firstUrl;
//     }
//   }

//   const prompts = chunks.map((chunk) =>
//     learningPromptTemplate
//       .replace("{{query}}", query)
//       .replace("{{title}}", derivedTitle)
//       .replace("{{url}}", typeof firstUrl === 'string' ? firstUrl : "No URL")
//       .replace("{{content}}", chunk)
//   );

//   const batchResults = await generateBatchWithTools(prompts, [{ googleSearch: {} }]);
//   const learnings: string[] = [];
//   for (const br of batchResults) {
//     const text = await br.response.text();
//     const parsedResult = z.object({
//       learnings: z.array(z.string()),
//       followUpQuestions: z.array(z.string()),
//     }).safeParse(JSON.parse(text));
//     if (parsedResult.success) {
//       learnings.push(...(parsedResult.data.learnings ?? []));
//     }
//   }

//   // --- START: ADDED LOGGING CODE ---
//   // Store the final return value in a variable
//   const returnValue = { learnings: learnings.slice(0, numLearnings) ?? [], followUpQuestions: [] };

//   // Prepare the output data for logging
//   const outputLog = `
// --- OUTPUT ---
// ${JSON.stringify(returnValue, null, 2)}
// ---------------------------------------------
// `;

//   // Append both the input and output logs to the file
//   fs.appendFileSync(logFilePath, inputLog + outputLog);

//   // Return the stored value
//   return returnValue;
//   // --- END: ADDED LOGGING CODE ---
// }

// Insert helper functions before writeFinalReport
// New helper function to generate an outline from the prompt and learnings (schema-based)
async function generateOutline(prompt: string, learnings: string[]): Promise<string> {
  try {
    const { OutlineJsonSchema: schema } = await import('./types.js');
    const outlinePrompt = `${systemPrompt()}\n\nBased on the prompt and the following learnings, generate a detailed outline as JSON with an 'outline' array.\nPrompt: ${prompt}\nLearnings:\n${learnings.join("\\n")}`;
    const json = await callGeminiProConfigurable(outlinePrompt, { json: true, schema, tools: [{ googleSearch: {} }] });
    let parsed: { outline?: string[] } = {};
    try {
      parsed = JSON.parse(json) as { outline?: string[] };
    } catch {}
    const outline = Array.isArray(parsed.outline) ? parsed.outline : [];
    return outline.length ? outline.join('\n') : 'Outline could not be generated.';
  } catch (error) {
    output.log('Error in generateOutline:', { error });
    return 'Outline could not be generated.';
  }
}

// New helper function to write a report from the generated outline and learnings (schema-based)
async function writeReportFromOutline(outline: string, learnings: string[]): Promise<string> {
  const cleanOutline = sanitizeReportContent(outline);
  const cleanLearnings = learnings.map(sanitizeReportContent);
  try {
    const { SectionsJsonSchema: schema } = await import('./types.js');
    const reportPrompt = `${systemPrompt()}\n\nUsing the following outline and learnings, produce a JSON with 'sections' (array of markdown strings) and optional 'citations'.\nOutline:\n${cleanOutline}\nLearnings:\n${cleanLearnings.join("\\n")}`;
    const json = await callGeminiProConfigurable(reportPrompt, { json: true, schema, tools: [{ googleSearch: {} }] });
    let parsed: { sections?: string[]; citations?: string[] } = {};
    try {
      parsed = JSON.parse(json) as { sections?: string[]; citations?: string[] };
    } catch {}
    const body = (parsed.sections ?? []).join('\n\n');
    const citations = parsed.citations?.length ? `\n\nReferences:\n${parsed.citations.map(c => `- ${c}`).join('\n')}` : '';
    return body || `Report could not be generated.${citations}`;
  } catch (error) {
    output.log('Error in writeReportFromOutline:', { error });
    return 'Report could not be generated.';
  }
}

// New helper function to generate a summary from the learnings (schema-based)
async function generateSummary(learnings: string[]): Promise<string> {
  try {
    const { SummaryJsonSchema: schema } = await import('./types.js');
    const summaryPrompt = `${systemPrompt()}\n\nReturn JSON with a single 'summary' string for the following learnings:\nLearnings:\n${learnings.join("\\n")}`;
    const json = await callGeminiProConfigurable(summaryPrompt, { json: true, schema, tools: [{ googleSearch: {} }] });
    let parsed: { summary?: string } = {};
    try {
      parsed = JSON.parse(json) as { summary?: string };
    } catch {}
    return typeof parsed.summary === 'string' ? parsed.summary : 'Summary could not be generated.';
  } catch (error) {
    output.log('Error in generateSummary:', { error });
    return 'Summary could not be generated.';
  }
}

// New helper function to generate a title from the prompt and learnings
async function generateTitle(prompt: string, learnings: string[]): Promise<string> {
  try {
    const { TitleJsonSchema: schema } = await import('./types.js');
    const titlePrompt = `${systemPrompt()}\n\nReturn JSON with a single 'title' for a research report based on the prompt and learnings:\nPrompt: ${prompt}\nLearnings:\n${learnings.join("\\n")}`;
    const json = await callGeminiProConfigurable(titlePrompt, { json: true, schema, tools: [{ googleSearch: {} }] });
    let parsed: { title?: string } = {};
    try { parsed = JSON.parse(json) as { title?: string }; } catch {}
    return typeof parsed.title === 'string' ? parsed.title : 'Untitled Research Report';
  } catch (error) {
    output.log('Error in generateTitle:', { error });
    return 'Title could not be generated.';
  }
}

interface DeepResearchOptions {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
  onProgress?: (progress: ResearchProgress) => void;
  reportProgress?: (progress: ResearchProgress) => void;
  initialQuery?: string;
  researchGoal?: string;
}

const DEFAULT_DEPTH = 2;
const DEFAULT_BREADTH = 5;

async function deepResearch({
  query,
  depth = DEFAULT_DEPTH,
  breadth = DEFAULT_BREADTH,
  learnings: initialLearnings = [],
  visitedUrls: initialVisitedUrls = [],
  onProgress,
  reportProgress = (progress: ResearchProgress) => {
    output.log('Research Progress:', progress);
  },
  initialQuery = query,
  researchGoal = "Deep dive research",
}: DeepResearchOptions): Promise<ResearchResult> {
  let visitedUrls = [...initialVisitedUrls];
  let learnings = [...initialLearnings];
  // Initialize progress object
  let progress: ResearchProgress = {
    currentDepth: depth,
    totalDepth: depth,
    currentBreadth: breadth,
    totalBreadth: breadth,
    totalQueries: breadth * depth,
    completedQueries: 0,
  };

  if (depth <= 0) {
    output.log("Reached research depth limit.");
    return { content: '', sources: [], methodology: '', limitations: '', citations: [], learnings: [], visitedUrls: [], firecrawlResults: { metadata: {} } as SearchResponse, analysis: '' };
  }

  if (visitedUrls.length > 20) {
    output.log("Reached visited URLs limit.");
    return { content: '', sources: [], methodology: '', limitations: '', citations: [], learnings: [], visitedUrls: [], firecrawlResults: { metadata: {} } as SearchResponse, analysis: '' };
  }

  const serpQueries = await generateSerpQueries({
    query,
    numQueries: breadth,
    learnings,
    researchGoal,
    initialQuery,
    depth,
    breadth,
  });

  const limit = pLimit(ConcurrencyLimit);
  const limitedProcessResult = async (serpQuery: SerpQuery): Promise<ProcessResult> => {
    output.log(`Processing serp query: ${serpQuery.query}...`);
    let newLearnings: string[] = [];
    let newUrls: string[] = [];
    try {
      output.log(`Generating Gemini prompt for query: ${serpQuery.query}...`);
      const prompt = generateGeminiPrompt({ query: serpQuery.query, researchGoal: serpQuery.researchGoal, learnings });
      output.log(`Gemini Prompt: ${prompt.substring(0, 200)}...`); // Log first 200 chars of prompt
      const geminiResponseText = await callGeminiProConfigurable(prompt, { tools: [{ googleSearch: {} }] });
      const geminiResult = await processGeminiResponse(geminiResponseText);
      newLearnings = geminiResult.learnings;
      newUrls = geminiResult.urls;

      if (visitedUrls.includes(serpQuery.query)) {
        output.log(`Already visited URL for query: ${serpQuery.query}, skipping.`);

        return {
          analysis: '',
          content: '',
          sources: [],
          methodology: '',
          limitations: '',
          citations: [],
          learnings: [],
          visitedUrls: [],
          firecrawlResults: { metadata: {} } as SearchResponse
        }; // Return empty result to avoid affecting overall learnings
      }

      try {
        output.log(`Firecrawl scraping for query: ${serpQuery.query}...`);
        // Firecrawl disabled in Gemini-only mode
        const result = { data: [] };
        // Add type assertion to result to ensure TypeScript knows the structure
        const firecrawlResult = result as { data?: Array<{ url?: string }> };
        if (!firecrawlResult || !firecrawlResult.data) {
          output.log(`Invalid Firecrawl result for query: ${serpQuery.query}`);
          return {
            analysis: '',
            content: '',
            sources: [],
            methodology: '',
            limitations: '',
            citations: [],
            learnings: [],
            visitedUrls: [],
            firecrawlResults: { metadata: {} } as SearchResponse
          };
        }

        // Collect URLs from this search, using optional chaining for safety
        newUrls = compact(firecrawlResult.data.map(item => item.url));
        const newBreadth = Math.ceil(breadth / 2);
        const newDepth = depth - 1;

        output.log("Researching deeper...");
        const processResult = await processSerpResult({
          query: serpQuery.query,
          result: { ...result, metadata: { success: true, error: '' } } as SearchResponse,
          numLearnings: 3,
        });
        newLearnings = processResult?.learnings ?? []; // Assign to the outer variable to avoid shadowing

        const allLearnings = [...learnings, ...newLearnings];
        const allUrls = [...visitedUrls, ...newUrls];

        if (newDepth > 0) {
          output.log(
            `Researching deeper, breadth: ${newBreadth}, depth: ${newDepth}`,
          );

          progress = {
            currentDepth: newDepth,
            currentBreadth: newBreadth,
            completedQueries: progress.completedQueries + 1,
            currentQuery: serpQuery.query,
            totalDepth: progress.totalDepth,
            totalBreadth: progress.totalBreadth,
            totalQueries: progress.totalQueries,
          };

          if (onProgress) {
            onProgress(progress);
          }
          // Structured local progress log as well
          logResearchProgress(progress);

          // Enhanced Query Refinement
          // const keywords = extractKeywords(query + " " + learnings.join(" ")); // Keyword extraction and query expansion functions are not defined in the provided code.
          // const refinedQuery = expandQuery(serpQuery.researchGoal, keywords);
          // const nextQuery = refinedQuery; // Using original query for now as refinement is not implemented
          const nextQuery = serpQuery.query; // Using original query for now as refinement is not implemented
          const deeper = await deepResearch({
            query: nextQuery,
            breadth: newBreadth,
            depth: newDepth,
            learnings: allLearnings,
            visitedUrls: allUrls,
            onProgress,
            reportProgress,
            initialQuery,
            researchGoal,
          });
          return {
            analysis: deeper.analysis,
            content: deeper.content,
            sources: deeper.sources,
            methodology: deeper.methodology,
            limitations: deeper.limitations,
            citations: deeper.citations.map(c => c.reference),
            learnings: deeper.learnings,
            visitedUrls: deeper.visitedUrls,
            firecrawlResults: deeper.firecrawlResults,
          };
        } else {
          output.log("Reached maximum research depth.");
          return {
            analysis: geminiResult.analysis || 'No analysis available',
            content: newLearnings.join('\n\n'),
            sources: [],
            methodology: 'Semantic chunking with Gemini Flash 2.5',
            limitations: 'Current implementation focuses on text analysis only',
            citations: [],
            learnings: newLearnings,
            visitedUrls: newUrls,
            firecrawlResults: { metadata: { success: false, error: 'No metadata' }, ...result } as unknown as SearchResponse,
          };
        }
      } catch (error) {
        output.log(`Error processing query ${serpQuery.query}: ${error}`);
        return {
          analysis: '',
          content: '',
          sources: [],
          methodology: '',
          limitations: '',
          citations: [],
          learnings: [],
          visitedUrls: [],
          firecrawlResults: { metadata: {} } as SearchResponse
        };
      } finally {
        progress.completedQueries += 1; // Increment completed queries count
        if (reportProgress) {
          reportProgress(progress); // Report progress after each query
        }
      }
    } catch (error) {
      output.log(`Error processing query ${serpQuery.query}: ${error}`);
      return {
        analysis: '',
        content: '',
        sources: [],
        methodology: '',
        limitations: '',
        citations: [],
        learnings: [],
        visitedUrls: [],
        firecrawlResults: { metadata: {} } as SearchResponse
      };
    }
  };

  const promises = serpQueries.map((serpQuery) => limit(() => limitedProcessResult(serpQuery)));

  const results = await Promise.all(promises);

  visitedUrls = compact(results.flatMap((result: ProcessResult) => result?.visitedUrls));
  learnings = compact(results.flatMap((result: ProcessResult) => result?.learnings));

  const processedData = {
    analysis: '',
    content: learnings.join('\n\n'),
    sources: [],
    methodology: 'Semantic chunking with Gemini Flash 2.5',
    limitations: 'Current implementation focuses on text analysis only',
    citations: [],
    learnings: learnings,
    visitedUrls: visitedUrls,
    firecrawlResults: { metadata: { success: false, error: 'Firecrawl disabled' }, data: [] } as SearchResponse,
  };

  // Firecrawl disabled in Gemini-only mode
  const firecrawlResponse = { data: [], metadata: { success: false, error: 'Firecrawl disabled' } } as SearchResponse;

  return {
    ...processedData,
    firecrawlResults: firecrawlResponse,
  };
}

interface WriteFinalReportParams {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}

export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
}: WriteFinalReportParams): Promise<string> {
  // 1. Create cache key
  let cacheKey: string;
  try {
    const keyObject: Record<string, unknown> = { prompt };
    const learningsHash = learnings.length > 0 ? String(learnings.reduce((acc, val) => acc + val.charCodeAt(0), 0)) : ''; // Hash learnings
    keyObject.learningsHash = learningsHash;
    const visitedUrlsHash = visitedUrls.length > 0 ? String(visitedUrls.reduce((acc, val) => acc + val.charCodeAt(0), 0)) : ''; // Hash visitedUrls (optional)
    keyObject.visitedUrlsHash = visitedUrlsHash; // Include visitedUrls hash in key (optional)
    cacheKey = JSON.stringify(keyObject);
  } catch (keyError) {
    output.log("Error creating report cache key:", { error: keyError instanceof Error ? keyError.message : 'Unknown error' });
    cacheKey = 'default-report-key'; // Fallback key
  }

  // 2. Check cache
  try {
    const cachedReport = reportCache.get(cacheKey);
    if (cachedReport) {
      output.log(`Returning cached report for key: ${cacheKey}`);
      return cachedReport.__returned;
    }
  } catch (cacheGetError) {
    output.log("Error getting report from cache:", { error: cacheGetError instanceof Error ? cacheGetError.message : 'Unknown error' });
    // Continue without cache if error
  }

  output.log("Generating outline...");
  const outline = await generateOutline(prompt, learnings);
  output.log("Outline generated:", { outline });

  output.log("Writing report from outline...");
  const report = await writeReportFromOutline(outline, learnings);
  output.log("Report generated.");

  output.log("Generating summary...");
  const summary = await generateSummary(learnings);
  output.log("Summary generated.");

  output.log("Generating title...");
  const title = await generateTitle(prompt, learnings);
  output.log("Title generated:", { title });

  const finalReport = `
# ${title}

## Abstract
${summary}

## Table of Contents
${outline}

## Introduction
This report investigates the query: "${prompt}" using grounded web research with Gemini. The following sections synthesize findings, with citations inline where provided.

## Body
${report}

## Methodology
Semantic chunking, grounded generation (Google Search tool), and schema-guided synthesis. Learnings were extracted from source content, deduplicated, and compiled.

## Limitations
Automated extraction may miss nuance or context from sources. Some links may be unavailable or rate-limited at retrieval time.

## Key Learnings
${learnings.map(learning => `- ${learning}`).join('\n')}

## References
${visitedUrls.map(url => `- ${url}`).join('\n')}
`;

  // const finalReportContent = await trimPrompt(finalReport, systemPrompt.length);
  //console.log("================\nFINAL PROMPT FOR REPORT:\n================\n", finalReport);
  const finalReportContent=finalReport;

  output.log("Final Research Report:", { finalReportContent });

  // 3. Store report in cache
  try {
    reportCache.set(cacheKey, {
      __returned: finalReportContent,
      __abortController: new AbortController(),
      __staleWhileFetching: undefined
    });
    output.log(`Cached report for key: ${cacheKey}`);
  } catch (cacheSetError) {
    output.log("Error setting report to cache:", { error: cacheSetError instanceof Error ? cacheSetError.message : 'Unknown error' });
  }

  output.saveResearchReport(finalReportContent);

  return finalReportContent;
}

export async function research(options: ResearchOptions): Promise<ResearchResult> {
  output.log(`Starting research for query: ${options.query}`);
  const researchResult = await deepResearch({
    query: options.query,
    depth: options.depth,
    breadth: options.breadth,
    learnings: options.existingLearnings || [],
    onProgress: options.onProgress,
  });
  output.log("Deep research completed. Generating final report...");

  const finalReport = await writeFinalReport({
    prompt: options.query,
    learnings: researchResult.learnings,
    visitedUrls: researchResult.visitedUrls,
  });
  output.log("Final report written. Research complete.");
  output.log(`Final Report of DeepResearch: ${finalReport}`); // Log the final report

  return researchResult;
}

export interface ResearchOptions {
    query: string;
    depth: number;
    breadth: number;
    existingLearnings?: string[];
    onProgress?: (progress: ResearchProgress) => void;
}

interface GeminiItem {
  learning?: string;
  learnings?: string[];
  url?: string;
}
interface GeminiResponse {
  items: GeminiItem[];
}

interface ProcessedGeminiResponse {
    analysis: string;
    learnings: string[];
    urls: string[];
    citations: { reference: string; context: string }[];
}


async function processGeminiResponse(geminiResponseText: string): Promise<ProcessedGeminiResponse> {
  const responseData = safeParseJSON<GeminiResponse>(geminiResponseText, { items: [] });

  const learnings: string[] = [];
  const urls: string[] = [];
  let sanitizedLearnings: string[] | undefined;
  if (Array.isArray(responseData.items)) {
    responseData.items.forEach(item => {
      // Add strict filtering for final research content
      if (item?.learning && typeof item.learning === 'string' &&
          !item.learning.includes('INTERNAL PROCESS:') &&
          !item.learning.startsWith('OUTLINE:') &&
          !item.learning.match(/^Step \d+:/)) {
        learnings.push(item.learning.trim());
      }
      if (item?.learnings && item.learnings.length > 0) {
        learnings.push(...item.learnings);
      }
      // Keep URL extraction as-is
      if (item?.url && typeof item.url === 'string') {
        urls.push(item.url.trim());
      }
    });

    // Add final sanitization before return
    sanitizedLearnings = learnings.map(l => l.replace(/\[.*?\]/g, '').trim()).filter(l => l.length > 0);
  }



  // Extract citations from response text
  const citationMatches = geminiResponseText.match(/\[\[\d+\]\]/g) || [];
  const citations = citationMatches.map(match => ({
    reference: match,
    context: geminiResponseText.split(match)[1]?.split('.')[0] || '' // Get first sentence after citation
  }));
  return {
    analysis: '',
    learnings: (sanitizedLearnings ?? learnings).slice(0, 10),
    urls: Array.from(new Set(urls)),
    citations
  };
}

export function validateAcademicInput(input: string): boolean {
  return input.length > 10 && input.trim().split(/\s+/).length >= 3;
}

export function validateAcademicOutput(text: string): boolean {
  const citationDensity = (text.match(/\[\[\d+\]\]/g) || []).length / (text.split(/\s+/).length / 100);
  const recentSources = (text.match(/\[\[\d{4}\]\]/g) || [])
    .filter((yr: string) => Number.parseInt(yr.replace(/[^\d]/g, ''), 10) > 2019).length;
  const conflictDisclosures = text.includes('Conflict Disclosure:') ? 1 : 0;
  return citationDensity > 1.5 && recentSources > 3 && conflictDisclosures === 1;
}

// Update conductResearch to use real sources
export async function conductResearch(
  query: string,
  depth: number = 3
): Promise<ResearchResult> {
  const splitter = new SemanticTextSplitter();
  const chunks = await splitter.splitText(query);
  // Use depth to limit how many chunks we process in this pass
  const limitedChunks = chunks.slice(0, Math.max(1, depth));
  // Batch model calls for all chunks
  const prompts = limitedChunks.map((chunk: string) => ({
    contents: [{ role: 'user', parts: [{ text: chunk }] }]
  }));
  const batch = await generateBatch(prompts);
  const results = await Promise.all(batch.map(b => b.response.text()));
  // Firecrawl disabled in Gemini-only mode
  const firecrawlResponse = { data: [], metadata: { success: false, error: 'Firecrawl disabled' } } as SearchResponse;
  return {
    content: results.join('\n\n'),
    sources: [],
    methodology: 'Semantic chunking with Gemini Flash 2.0',
    limitations: 'Current implementation focuses on text analysis only',
    citations: results.flatMap((r: string) =>
      (r.match(/\[\[\d+\]\]/g) || []).map((ref: string) => ({ reference: ref }))
    ), // Extract citations from content
    learnings: [],
    visitedUrls: [],
    firecrawlResults: firecrawlResponse,
    analysis: ''
  };
}

// Create proper empty ResearchResult objects
const createEmptyResearchResult = (): ResearchResult => ({
  content: '',
  sources: [],
  methodology: '',
  limitations: '',
  citations: [],
  learnings: [],
  visitedUrls: [],
  firecrawlResults: { metadata: {} } as SearchResponse,
  analysis: ''
});

// Add type-safe firecrawl result handling
interface FirecrawlResult {
  url?: string;
  markdown?: string | Promise<string>;
}

// Update the SearchResponse interface to include metadata
interface SearchResponse {
  data: FirecrawlResult[];
  metadata: { success: boolean; error: string }; // Add metadata property
}

// Type guard helpers
const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null;

// Then use in processFirecrawlData:
const processFirecrawlData = (result: unknown): SearchResponse => {
  const dataRaw = isObject(result) && Array.isArray(result.data) ? (result.data as unknown[]) : [];
  const data = dataRaw.filter((item: unknown): item is FirecrawlResult => isObject(item) && typeof item.url === 'string');
  const metadata = isObject(result) && isObject(result.metadata)
    ? (result.metadata as { success?: unknown; error?: unknown })
    : {};
  const success = typeof (metadata.success) === 'boolean' ? metadata.success : false;
  const error = typeof (metadata.error) === 'string' ? metadata.error : 'No metadata';
  return { data, metadata: { success, error } };
};
