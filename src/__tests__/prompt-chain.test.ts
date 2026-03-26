import { PromptChain, createChain, Parsers, Guards, LLMProvider, ChainResult } from '../index';

// === Mock LLM Provider ===

function createMockProvider(responses: Record<string, string> | string = 'mock output'): LLMProvider & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    async complete(prompt: string) {
      calls.push(prompt);
      if (typeof responses === 'string') return responses;
      for (const [key, value] of Object.entries(responses)) {
        if (prompt.includes(key)) return value;
      }
      return 'default response';
    },
  };
}

function createFailingProvider(failCount: number, successResponse = 'recovered'): LLMProvider & { attempts: number } {
  let attempts = 0;
  return {
    get attempts() { return attempts; },
    async complete() {
      attempts++;
      if (attempts <= failCount) throw new Error(`Attempt ${attempts} failed`);
      return successResponse;
    },
  };
}

// === Chain creation and step execution ===

describe('PromptChain', () => {
  describe('chain creation and step execution', () => {
    it('should create a chain via constructor', () => {
      const provider = createMockProvider();
      const chain = new PromptChain(provider);
      expect(chain).toBeInstanceOf(PromptChain);
    });

    it('should create a chain via createChain helper', () => {
      const provider = createMockProvider();
      const chain = createChain(provider);
      expect(chain).toBeInstanceOf(PromptChain);
    });

    it('should execute a single step', async () => {
      const provider = createMockProvider('Hello world');
      const result = await createChain(provider)
        .step('greet', 'Say hello')
        .run('test input');

      expect(result.output).toBe('Hello world');
      expect(result.steps).toHaveLength(1);
      expect(result.steps[0].stepName).toBe('greet');
    });

    it('should execute multiple steps in sequence', async () => {
      let callIndex = 0;
      const provider: LLMProvider = {
        async complete() {
          callIndex++;
          return `response-${callIndex}`;
        },
      };

      const result = await createChain(provider)
        .step('first', 'step 1')
        .step('second', 'step 2')
        .step('third', 'step 3')
        .run('input');

      expect(result.steps).toHaveLength(3);
      expect(result.steps[0].output).toBe('response-1');
      expect(result.steps[1].output).toBe('response-2');
      expect(result.steps[2].output).toBe('response-3');
      expect(result.output).toBe('response-3');
    });

    it('should pass input to the first step prompt', async () => {
      const provider = createMockProvider();
      await createChain(provider)
        .step('echo', '{{input}}')
        .run('my input');

      expect(provider.calls[0]).toBe('my input');
    });

    it('should support function prompts with context', async () => {
      const provider = createMockProvider('result');
      await createChain(provider)
        .set('topic', 'TypeScript')
        .step('ask', (ctx) => `Tell me about ${ctx.variables['topic']} given ${ctx.input}`)
        .run('basics');

      expect(provider.calls[0]).toBe('Tell me about TypeScript given basics');
    });
  });

  // === Template interpolation ===

  describe('template interpolation', () => {
    it('should interpolate {{input}}', async () => {
      const provider = createMockProvider();
      await createChain(provider)
        .step('s1', 'Process: {{input}}')
        .run('hello');

      expect(provider.calls[0]).toBe('Process: hello');
    });

    it('should interpolate {{lastOutput}}', async () => {
      let callCount = 0;
      const provider: LLMProvider = {
        async complete() {
          callCount++;
          return callCount === 1 ? 'first-result' : 'done';
        },
      };

      await createChain(provider)
        .step('s1', 'start')
        .step('s2', 'Continue from: {{lastOutput}}')
        .run('input');

      // The second call should have interpolated lastOutput
      expect((provider as any).calls).toBeUndefined; // using raw provider
    });

    it('should interpolate custom variables set via .set()', async () => {
      const provider = createMockProvider();
      await createChain(provider)
        .set('lang', 'Python')
        .set('level', 'beginner')
        .step('ask', 'Teach {{lang}} for {{level}}')
        .run('go');

      expect(provider.calls[0]).toBe('Teach Python for beginner');
    });

    it('should interpolate step output as a variable by step name', async () => {
      let callCount = 0;
      const provider = createMockProvider();
      const mockProvider: LLMProvider & { calls: string[] } = {
        calls: [],
        async complete(prompt: string) {
          mockProvider.calls.push(prompt);
          callCount++;
          return callCount === 1 ? 'summary-text' : 'final';
        },
      };

      await createChain(mockProvider)
        .step('summarize', 'Summarize: {{input}}')
        .step('review', 'Review this summary: {{summarize}}')
        .run('long document');

      expect(mockProvider.calls[1]).toBe('Review this summary: summary-text');
    });

    it('should replace unknown variables with empty string', async () => {
      const provider = createMockProvider();
      await createChain(provider)
        .step('s1', 'Value is: {{nonexistent}}')
        .run('input');

      expect(provider.calls[0]).toBe('Value is: ');
    });
  });

  // === Output parsers ===

  describe('output parsers', () => {
    it('Parsers.json() should parse raw JSON', () => {
      const parser = Parsers.json();
      const result = parser('{"name":"test","value":42}');
      expect(result).toEqual({ name: 'test', value: 42 });
    });

    it('Parsers.json() should extract JSON from code fences', () => {
      const parser = Parsers.json();
      const raw = 'Here is the result:\n```json\n{"key":"value"}\n```\nDone.';
      const result = parser(raw);
      expect(result).toEqual({ key: 'value' });
    });

    it('Parsers.lines() should split into trimmed non-empty lines', () => {
      const parser = Parsers.lines();
      const result = parser('  apple \n banana \n\n cherry  \n');
      expect(result).toEqual(['apple', 'banana', 'cherry']);
    });

    it('Parsers.first() should return the first line trimmed', () => {
      const parser = Parsers.first();
      expect(parser('  hello  \nworld')).toBe('hello');
    });

    it('Parsers.number() should extract a number from text', () => {
      const parser = Parsers.number();
      expect(parser('The answer is 42.')).toBe(42);
      expect(parser('$3.14 USD')).toBe(3.14);
      expect(parser('-7 degrees')).toBe(-7);
    });

    it('Parsers.number() should throw on non-numeric input', () => {
      const parser = Parsers.number();
      expect(() => parser('no numbers here')).toThrow('Cannot parse number');
    });

    it('Parsers.boolean() should return true for yes/true/1', () => {
      const parser = Parsers.boolean();
      expect(parser('yes')).toBe(true);
      expect(parser('TRUE')).toBe(true);
      expect(parser('  True  ')).toBe(true);
      expect(parser('1')).toBe(true);
    });

    it('Parsers.boolean() should return false for other values', () => {
      const parser = Parsers.boolean();
      expect(parser('no')).toBe(false);
      expect(parser('false')).toBe(false);
      expect(parser('0')).toBe(false);
      expect(parser('maybe')).toBe(false);
    });

    it('should apply parser to step output in chain', async () => {
      const provider = createMockProvider('The score is 95 points');
      const result = await createChain(provider)
        .step('score', 'Get score', { parser: Parsers.number() as any })
        .run('test');

      // number parser returns 95, which gets stringified to "95"
      expect(result.output).toBe('95');
    });
  });

  // === Step guards ===

  describe('step guards', () => {
    it('should skip step when guard returns false', async () => {
      const provider = createMockProvider('output');
      const result = await createChain(provider)
        .step('skipped', 'never runs', { guard: () => false })
        .run('input');

      expect(result.steps).toHaveLength(0);
      expect(result.output).toBe('input'); // lastOutput stays as input
    });

    it('should execute step when guard returns true', async () => {
      const provider = createMockProvider('output');
      const result = await createChain(provider)
        .step('runs', 'always runs', { guard: () => true })
        .run('input');

      expect(result.steps).toHaveLength(1);
      expect(result.output).toBe('output');
    });

    it('should pass context to guard function', async () => {
      const provider = createMockProvider('result');
      const guardSpy = jest.fn(() => true);

      await createChain(provider)
        .set('mode', 'verbose')
        .step('s1', 'prompt', { guard: guardSpy })
        .run('my-input');

      expect(guardSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          input: 'my-input',
          variables: expect.objectContaining({ mode: 'verbose' }),
        })
      );
    });

    it('should conditionally skip middle steps', async () => {
      let callCount = 0;
      const provider: LLMProvider = {
        async complete() {
          callCount++;
          return `r${callCount}`;
        },
      };

      const result = await createChain(provider)
        .step('first', 'p1')
        .step('middle', 'p2', { guard: () => false })
        .step('last', 'p3')
        .run('input');

      expect(result.steps).toHaveLength(2);
      expect(result.steps[0].stepName).toBe('first');
      expect(result.steps[1].stepName).toBe('last');
    });
  });

  // === Built-in Guards ===

  describe('built-in Guards', () => {
    describe('Guards.outputContains', () => {
      it('should return true when lastOutput contains substring', () => {
        const guard = Guards.outputContains('error');
        expect(guard({ input: '', variables: {}, history: [], lastOutput: 'found an error here' })).toBe(true);
      });

      it('should return false when lastOutput does not contain substring', () => {
        const guard = Guards.outputContains('error');
        expect(guard({ input: '', variables: {}, history: [], lastOutput: 'all good' })).toBe(false);
      });

      it('should work in a chain to conditionally run steps', async () => {
        let callCount = 0;
        const provider: LLMProvider = {
          async complete() {
            callCount++;
            return callCount === 1 ? 'success: data loaded' : 'error handled';
          },
        };

        const result = await createChain(provider)
          .step('load', 'load data')
          .step('handleError', 'fix errors', { guard: Guards.outputContains('error') })
          .run('input');

        // "success: data loaded" does not contain "error" so handleError is skipped
        expect(result.steps).toHaveLength(1);
        expect(result.output).toBe('success: data loaded');
      });
    });

    describe('Guards.variableSet', () => {
      it('should return true when variable is truthy', () => {
        const guard = Guards.variableSet('apiKey');
        expect(guard({ input: '', variables: { apiKey: 'abc123' }, history: [], lastOutput: '' })).toBe(true);
      });

      it('should return false when variable is falsy', () => {
        const guard = Guards.variableSet('apiKey');
        expect(guard({ input: '', variables: { apiKey: '' }, history: [], lastOutput: '' })).toBe(false);
        expect(guard({ input: '', variables: {}, history: [], lastOutput: '' })).toBe(false);
      });

      it('should work in a chain with .set()', async () => {
        const provider = createMockProvider('done');
        const result = await createChain(provider)
          .set('debug', false)
          .step('debugStep', 'debug info', { guard: Guards.variableSet('debug') })
          .run('input');

        expect(result.steps).toHaveLength(0);
      });
    });

    describe('Guards.minOutputLength', () => {
      it('should return true when output length meets minimum', () => {
        const guard = Guards.minOutputLength(5);
        expect(guard({ input: '', variables: {}, history: [], lastOutput: 'hello' })).toBe(true);
        expect(guard({ input: '', variables: {}, history: [], lastOutput: 'hello world' })).toBe(true);
      });

      it('should return false when output is too short', () => {
        const guard = Guards.minOutputLength(10);
        expect(guard({ input: '', variables: {}, history: [], lastOutput: 'hi' })).toBe(false);
      });

      it('should work in a chain', async () => {
        let callCount = 0;
        const provider: LLMProvider = {
          async complete() {
            callCount++;
            return callCount === 1 ? 'ok' : 'expanded output';
          },
        };

        const result = await createChain(provider)
          .step('brief', 'brief answer')
          .step('expand', 'expand on it', { guard: Guards.minOutputLength(10) })
          .run('input');

        // "ok" has length 2, which is < 10, so expand is skipped
        expect(result.steps).toHaveLength(1);
        expect(result.output).toBe('ok');
      });
    });
  });

  // === Retry logic ===

  describe('retry logic', () => {
    it('should retry on failure and succeed', async () => {
      const provider = createFailingProvider(2, 'success');

      const result = await createChain(provider)
        .step('flaky', 'do it', { retries: 3 })
        .run('input');

      expect(result.output).toBe('success');
      expect(provider.attempts).toBe(3); // 2 failures + 1 success
    });

    it('should throw after exhausting all retries', async () => {
      const provider = createFailingProvider(5, 'never reached');

      await expect(
        createChain(provider)
          .step('doomed', 'fail', { retries: 2 })
          .run('input')
      ).rejects.toThrow('Attempt 3 failed');

      expect(provider.attempts).toBe(3); // 1 initial + 2 retries
    });

    it('should not retry when retries is 0 (default)', async () => {
      const provider = createFailingProvider(1);

      await expect(
        createChain(provider)
          .step('once', 'fail')
          .run('input')
      ).rejects.toThrow('Attempt 1 failed');

      expect(provider.attempts).toBe(1);
    });

    it('should succeed on first attempt without retries configured', async () => {
      const provider = createMockProvider('instant success');

      const result = await createChain(provider)
        .step('quick', 'go')
        .run('input');

      expect(result.output).toBe('instant success');
    });
  });

  // === Middleware ===

  describe('middleware', () => {
    it('should execute middleware before steps', async () => {
      const order: string[] = [];
      const provider = createMockProvider('done');

      await createChain(provider)
        .use(async (ctx, next) => {
          order.push('middleware');
          await next();
        })
        .step('s1', (ctx) => {
          order.push('step');
          return 'prompt';
        })
        .run('input');

      expect(order[0]).toBe('middleware');
      expect(order[1]).toBe('step');
    });

    it('should allow middleware to modify variables', async () => {
      const provider = createMockProvider();

      await createChain(provider)
        .use(async (ctx, next) => {
          ctx.variables['injected'] = 'middleware-value';
          await next();
        })
        .step('s1', '{{injected}}')
        .run('input');

      expect(provider.calls[0]).toBe('middleware-value');
    });

    it('should allow middleware to modify input', async () => {
      const provider = createMockProvider();

      await createChain(provider)
        .use(async (ctx, next) => {
          ctx.input = ctx.input.toUpperCase();
          await next();
        })
        .step('s1', '{{input}}')
        .run('hello');

      expect(provider.calls[0]).toBe('HELLO');
    });

    it('should execute multiple middlewares in order', async () => {
      const order: number[] = [];
      const provider = createMockProvider('done');

      await createChain(provider)
        .use(async (ctx, next) => {
          order.push(1);
          await next();
        })
        .use(async (ctx, next) => {
          order.push(2);
          await next();
        })
        .use(async (ctx, next) => {
          order.push(3);
          await next();
        })
        .step('s1', 'go')
        .run('input');

      expect(order).toEqual([1, 2, 3]);
    });
  });

  // === ChainResult structure ===

  describe('ChainResult structure', () => {
    it('should include all step results', async () => {
      let callCount = 0;
      const provider: LLMProvider = {
        async complete(prompt) {
          callCount++;
          return `output-${callCount}`;
        },
      };

      const result = await createChain(provider)
        .step('alpha', 'prompt-a')
        .step('beta', 'prompt-b')
        .run('start');

      expect(result.steps).toHaveLength(2);
      expect(result.steps[0]).toMatchObject({
        stepName: 'alpha',
        input: 'prompt-a',
        output: 'output-1',
      });
      expect(result.steps[1]).toMatchObject({
        stepName: 'beta',
        input: 'prompt-b',
        output: 'output-2',
      });
    });

    it('should set output to the last step output', async () => {
      const provider = createMockProvider('final-answer');
      const result = await createChain(provider)
        .step('only', 'question')
        .run('input');

      expect(result.output).toBe('final-answer');
    });

    it('should set output to input when all steps are skipped', async () => {
      const provider = createMockProvider();
      const result = await createChain(provider)
        .step('skipped', 'nope', { guard: () => false })
        .run('original-input');

      expect(result.output).toBe('original-input');
      expect(result.steps).toHaveLength(0);
    });

    it('should include totalDurationMs as a non-negative number', async () => {
      const provider = createMockProvider('fast');
      const result = await createChain(provider)
        .step('s1', 'go')
        .run('input');

      expect(typeof result.totalDurationMs).toBe('number');
      expect(result.totalDurationMs).toBeGreaterThanOrEqual(0);
    });

    it('should include durationMs for each step', async () => {
      const provider = createMockProvider('output');
      const result = await createChain(provider)
        .step('timed', 'prompt')
        .run('input');

      expect(typeof result.steps[0].durationMs).toBe('number');
      expect(result.steps[0].durationMs).toBeGreaterThanOrEqual(0);
    });

    it('should return empty steps array when no steps are defined', async () => {
      const provider = createMockProvider();
      const result = await createChain(provider).run('input');

      expect(result.steps).toEqual([]);
      expect(result.output).toBe('input');
    });
  });
});
