/**
 * Compile a Zod decision schema into System One questions, and answers back into the schema.
 *
 * `z.enum([...])` compiles to a {@link Choice}, `z.boolean()` to a {@link YesNo}, and {@link score} (a branded
 * `z.number()`) to a {@link Score}. Descriptions come from `.describe()` or the marker helpers {@link choice},
 * {@link yesNo}, and {@link score}.
 */
import { z } from 'zod'

import { Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer } from './types.js'
import type { Answer, JSONContent, Question } from './types.js'

/** Option added to an optional Choice field; selecting it yields `undefined` (or `null` for a nullable field). */
export const NO_MATCH_OPTION = 'none'
const NO_MATCH_DESCRIPTION = 'None of the other options applies'
const GENERATE_HINT =
  'System One models select or score; they do not generate. Extract candidates in code and ask a Choice ' +
  'over them, or use an LLM for this field.'

/** A {@link score} field: a Zod number branded `DecisionScore`. */
export type ScoreField = z.core.$ZodBranded<z.ZodNumber, 'DecisionScore'>

/** A decision schema: a Zod object whose fields are enums, booleans, or {@link score} numbers. */
export type DecisionSchema = z.ZodObject

interface ChoiceMarker {
  readonly kind: 'choice'
  readonly instructions?: JSONContent
  readonly options: Readonly<Record<string, JSONContent | null>>
}

interface ScoreMarker {
  readonly kind: 'score'
  readonly instructions?: JSONContent
  readonly levels: readonly JSONContent[]
}

interface YesNoMarker {
  readonly kind: 'yesno'
  readonly instructions?: JSONContent
  readonly true?: JSONContent
  readonly false?: JSONContent
  readonly threshold?: number
}

type Marker = ChoiceMarker | ScoreMarker | YesNoMarker

const markers = z.registry<Marker>()

/**
 * Declare a Choice field: an enum over `options`, with optional per-option descriptions.
 *
 * @param options - Option names, or option name to description (null for none)
 * @param instructions - The decision to make; defaults to the field's `.describe()` text
 * @returns A Zod enum carrying the Choice marker
 * @throws Error if `options` is empty
 */
export function choice<const T extends readonly [string, ...string[]]>(
  options: T,
  instructions?: JSONContent
): z.ZodEnum<{ [K in T[number]]: K }>
export function choice<const T extends Readonly<Record<string, JSONContent | null>>>(
  options: T,
  instructions?: JSONContent
): z.ZodEnum<{ [K in keyof T & string]: K }>
export function choice(
  options: readonly string[] | Readonly<Record<string, JSONContent | null>>,
  instructions?: JSONContent
): z.ZodEnum {
  const described: Readonly<Record<string, JSONContent | null>> = Array.isArray(options)
    ? Object.fromEntries(options.map((option) => [option, null]))
    : (options as Readonly<Record<string, JSONContent | null>>)
  const names = Object.keys(described)
  if (names.length === 0) throw new Error('choice() needs at least one option')
  const schema = z.enum(names as [string, ...string[]])
  markers.add(schema, { kind: 'choice', options: described, ...(instructions !== undefined && { instructions }) })
  return schema
}

/**
 * Declare a Score field: a branded number rated along ordered, self-describing levels.
 *
 * @param levels - Ordered level descriptions, lowest first; level `i` scores `i`
 * @param instructions - What to rate; defaults to the field's `.describe()` text
 * @returns A branded Zod number carrying the Score marker
 * @throws Error if there are fewer than two levels
 */
export function score(levels: readonly JSONContent[], instructions?: JSONContent): ScoreField {
  // Validates the level count now, so a malformed schema fails where it is declared.
  new Score(instructions ?? 'score', levels)
  const schema = z.number().brand<'DecisionScore'>()
  markers.add(schema, { kind: 'score', levels, ...(instructions !== undefined && { instructions }) })
  return schema
}

/** Options for a {@link yesNo} field. */
export interface YesNoFieldOptions {
  /** Description of what a yes means. */
  readonly true?: JSONContent
  /** Description of what a no means. */
  readonly false?: JSONContent
  /** Probability at or above which the field reads true. Defaults to 0.5. */
  readonly threshold?: number
}

/**
 * Declare a YesNo field: a boolean that reads true when the answer's probability reaches `threshold`.
 *
 * @param instructions - The yes/no question; defaults to the field's `.describe()` text
 * @param options - Outcome descriptions and the boolean threshold
 * @returns A Zod boolean carrying the YesNo marker
 * @throws Error if threshold is outside [0, 1]
 */
export function yesNo(instructions?: JSONContent, options: YesNoFieldOptions = {}): z.ZodBoolean {
  new YesNo(instructions ?? 'yes/no', options)
  const schema = z.boolean()
  markers.add(schema, { kind: 'yesno', ...options, ...(instructions !== undefined && { instructions }) })
  return schema
}

type FieldKind = 'choice' | 'yesno' | 'score'

/** One schema field: the question to ask and how to turn its answer back into a field value. */
interface CompiledField {
  readonly name: string
  readonly question: Question
  readonly kind: FieldKind
  /** How a no-match Choice answer is represented, when the field is optional or nullable. */
  readonly noMatch?: null | undefined
  readonly optional: boolean
}

/** A decision schema compiled to questions. Compiled once per schema and reused for every request. */
export class CompiledSchema<S extends DecisionSchema = DecisionSchema> {
  /** The source schema. */
  readonly schema: S
  private readonly _fields: readonly CompiledField[]

  /**
   * @param schema - The source schema
   * @param fields - Its compiled fields
   * @internal
   */
  constructor(schema: S, fields: readonly CompiledField[]) {
    this.schema = schema
    this._fields = fields
  }

  /** Questions keyed by field name. */
  get questions(): Record<string, Question> {
    return Object.fromEntries(this._fields.map((field) => [field.name, field.question]))
  }

  /**
   * Map answers to a validated schema output.
   *
   * @param answers - Answers keyed by field name
   * @returns The parsed schema output
   * @throws Error if an answer is missing or has the wrong type for its field
   */
  buildOutput(answers: Readonly<Record<string, Answer>>): z.infer<S> {
    const values: Record<string, unknown> = {}
    for (const field of this._fields) {
      const answer = answers[field.name]
      if (answer === undefined) throw new Error(`decision model returned no answer for ${field.name}`)
      const value = fieldValue(field, answer)
      if (value !== undefined) values[field.name] = value
    }
    return this.schema.parse(values) as z.infer<S>
  }
}

const compiledCache = new WeakMap<DecisionSchema, CompiledSchema>()

/**
 * Compile `schema` into questions, caching the result per schema object.
 *
 * @param schema - A Zod object of enum, boolean, and {@link score} fields
 * @returns The compiled schema
 * @throws TypeError if `schema` is not a Zod object or a field cannot be asked of a System One model
 */
export function compileSchema<S extends DecisionSchema>(schema: S): CompiledSchema<S> {
  if (!(schema instanceof z.ZodObject)) throw new TypeError('decision schema must be a z.object(...)')
  const cached = compiledCache.get(schema)
  if (cached !== undefined) return cached as CompiledSchema<S>
  const entries = Object.entries(schema.shape as Record<string, z.ZodType>)
  if (entries.length === 0) throw new TypeError('decision schema has no fields to decide')
  const compiled = new CompiledSchema(
    schema,
    entries.map(([name, field]) => compileField(name, field))
  )
  compiledCache.set(schema, compiled)
  return compiled
}

function fieldValue(field: CompiledField, answer: Answer): unknown {
  const expected = { choice: ChoiceAnswer, yesno: YesNoAnswer, score: ScoreAnswer }[field.kind]
  if (!(answer instanceof expected)) {
    throw new Error(`${field.name}: expected a ${field.kind} answer, got ${answer.constructor.name}`)
  }
  if (answer instanceof ChoiceAnswer) {
    return field.optional && answer.choice === NO_MATCH_OPTION ? field.noMatch : answer.choice
  }
  if (answer instanceof YesNoAnswer) return answer.probability >= (field.question as YesNo).threshold
  return (answer as ScoreAnswer).score
}

interface Unwrapped {
  readonly inner: z.ZodType
  readonly optional: boolean
  readonly noMatch?: null | undefined
  readonly description?: string
}

/** Strip `.optional()` / `.nullable()` wrappers, keeping the outermost description. */
function unwrap(field: z.ZodType): Unwrapped {
  let inner = field
  let optional = false
  let noMatch: null | undefined = undefined
  const description = field.description
  while (inner instanceof z.ZodOptional || inner instanceof z.ZodNullable) {
    optional = true
    if (inner instanceof z.ZodNullable) noMatch = null
    inner = inner.unwrap() as z.ZodType
  }
  const innerDescription = description ?? inner.description
  return { inner, optional, noMatch, ...(innerDescription !== undefined && { description: innerDescription }) }
}

function compileField(name: string, field: z.ZodType): CompiledField {
  const { inner, optional, noMatch, description } = unwrap(field)
  const marker = markers.get(inner)
  if (inner instanceof z.ZodEnum) return compileChoice(name, inner, marker, description, optional, noMatch)
  if (inner instanceof z.ZodBoolean) return compileYesNo(name, marker, description, optional)
  if (inner instanceof z.ZodNumber && marker?.kind === 'score') {
    if (optional)
      throw new TypeError(`${name}: a score field cannot be optional; every level set always yields a score`)
    return {
      name,
      kind: 'score',
      optional: false,
      question: new Score(instructions(name, marker, description), marker.levels),
    }
  }
  if (inner instanceof z.ZodNumber) throw new TypeError(`${name}: a number field needs score([...levels])`)
  throw new TypeError(`${name}: unsupported field type '${inner.def.type}'. ${GENERATE_HINT}`)
}

function compileChoice(
  name: string,
  schema: z.ZodEnum,
  marker: Marker | undefined,
  description: string | undefined,
  optional: boolean,
  noMatch: null | undefined
): CompiledField {
  if (marker !== undefined && marker.kind !== 'choice') {
    throw new TypeError(`${name}: an enum field takes a choice() marker, not ${marker.kind}`)
  }
  const names = schema.options
  if (!names.every((value) => typeof value === 'string')) {
    throw new TypeError(`${name}: Choice options must be strings`)
  }
  if (optional && names.includes(NO_MATCH_OPTION)) {
    throw new TypeError(`${name}: an optional Choice reserves the '${NO_MATCH_OPTION}' option for no-match`)
  }
  const described = marker?.options ?? {}
  const options: Record<string, JSONContent | null> = Object.fromEntries(
    names.map((option) => [option, described[option as string] ?? null])
  )
  if (optional) options[NO_MATCH_OPTION] = NO_MATCH_DESCRIPTION
  return {
    name,
    kind: 'choice',
    optional,
    noMatch,
    question: new Choice(instructions(name, marker, description), options),
  }
}

function compileYesNo(
  name: string,
  marker: Marker | undefined,
  description: string | undefined,
  optional: boolean
): CompiledField {
  if (optional) throw new TypeError(`${name}: a yes/no field cannot be optional; it always yields a probability`)
  if (marker !== undefined && marker.kind !== 'yesno') {
    throw new TypeError(`${name}: a boolean field takes a yesNo() marker, not ${marker.kind}`)
  }
  const { instructions: _instructions, kind: _kind, ...options } = marker ?? { kind: 'yesno' }
  return {
    name,
    kind: 'yesno',
    optional: false,
    question: new YesNo(instructions(name, marker, description), options),
  }
}

function instructions(name: string, marker: Marker | undefined, description: string | undefined): JSONContent {
  const text = marker?.instructions ?? description
  if (text !== undefined && text !== '') return text
  throw new TypeError(`${name}: give the field instructions via its marker or .describe(...)`)
}
