// === Types ===

export interface LLMProvider {
  complete(prompt: string, options?: CompletionOptions): Promise<string>;
}

export interface CompletionOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stop?: string[];
}

export interface StepResult {
  stepName: string;
  input: string;
  output: string;
  durationMs: number;
  tokens?: { prompt: number; completion: number };
}

export interface ChainResult {
  output: string;
  steps: StepResult[];
  totalDurationMs: number;
}

type PromptTemplate = string | ((context: ChainContext) => string);
type OutputParser<T = string> = (raw: string) => T;
type StepGuard = (context: ChainContext) => boolean;

export interface ChainContext {
  input: string;
  variables: Record<string, unknown>;
  history: StepResult[];
  lastOutput: string;
}

interface ChainStep {
  name: string;
  prompt: PromptTemplate;
  options?: CompletionOptions;
  parser?: OutputParser;
  guard?: StepGuard;
  retries?: number;
}

// === Prompt Chain ===

export class PromptChain {
  private steps: ChainStep[] = [];
  private variables: Record<string, unknown> = {};
  private middlewares: Array<(ctx: ChainContext, next: () => Promise<void>) => Promise<void>> = [];

  constructor(private provider: LLMProvider) {}

  step(
    name: string,
    prompt: PromptTemplate,
    options?: CompletionOptions & { parser?: OutputParser; guard?: StepGuard; retries?: number }
  ): this {
    const { parser, guard, retries, ...completionOptions } = options || {};
    this.steps.push({ name, prompt, options: completionOptions, parser, guard, retries });
    return this;
  }

  set(key: string, value: unknown): this {
    this.variables[key] = value;
    return this;
  }

  use(middleware: (ctx: ChainContext, next: () => Promise<void>) => Promise<void>): this {
    this.middlewares.push(middleware);
    return this;
  }

  async run(input: string): Promise<ChainResult> {
    const startTime = Date.now();
    const results: StepResult[] = [];

    const context: ChainContext = {
      input,
      variables: { ...this.variables },
      history: results,
      lastOutput: input,
    };

    // Run middleware
    for (const mw of this.middlewares) {
      await mw(context, async () => {});
    }

    for (const step of this.steps) {
      // Check guard condition
      if (step.guard && !step.guard(context)) {
        continue;
      }

      const resolvedPrompt = typeof step.prompt === 'function'
        ? step.prompt(context)
        : this.interpolate(step.prompt, context);

      const stepStart = Date.now();
      let output: string | undefined;
      let lastError: Error | undefined;

      const attempts = (step.retries ?? 0) + 1;
      for (let attempt = 0; attempt < attempts; attempt++) {
        try {
          output = await this.provider.complete(resolvedPrompt, step.options);
          break;
        } catch (err) {
          lastError = err as Error;
        }
      }

      if (output === undefined) {
        throw lastError ?? new Error(`Step "${step.name}" failed`);
      }

      const parsed = step.parser ? String(step.parser(output)) : output;

      const result: StepResult = {
        stepName: step.name,
        input: resolvedPrompt,
        output: parsed,
        durationMs: Date.now() - stepStart,
      };

      results.push(result);
      context.lastOutput = parsed;
      context.variables[step.name] = parsed;
    }

    return {
      output: context.lastOutput,
      steps: results,
      totalDurationMs: Date.now() - startTime,
    };
  }

  private interpolate(template: string, context: ChainContext): string {
    return template.replace(/\{\{(\w+)\}\}/g, (_, key) => {
      if (key === 'input') return context.input;
      if (key === 'lastOutput') return context.lastOutput;
      return String(context.variables[key] ?? '');
    });
  }
}

// === Built-in Parsers ===

export const Parsers = {
  json: <T = unknown>(): OutputParser<T> => (raw: string) => {
    const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/) || [null, raw];
    return JSON.parse(match[1]!.trim());
  },

  lines: (): OutputParser<string[]> => (raw: string) =>
    raw.split('\n').map(l => l.trim()).filter(Boolean) as unknown as string[] & string,

  first: (): OutputParser => (raw: string) =>
    raw.split('\n')[0]?.trim() ?? '',

  number: (): OutputParser<number> => (raw: string) => {
    const num = parseFloat(raw.replace(/[^0-9.-]/g, ''));
    if (isNaN(num)) throw new Error(`Cannot parse number from: ${raw}`);
    return num as unknown as number & string;
  },

  boolean: (): OutputParser<boolean> => (raw: string) => {
    const lower = raw.toLowerCase().trim();
    return (['yes', 'true', '1'].includes(lower)) as unknown as boolean & string;
  },
};

// === Helper to create chain ===

export function createChain(provider: LLMProvider): PromptChain {
  return new PromptChain(provider);
}

export default PromptChain;
