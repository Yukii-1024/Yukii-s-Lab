const WORDS_PER_MINUTE_CN = 300;
const WORDS_PER_MINUTE_EN = 200;

export function getReadingTime(text: string): number {
	const cleaned = text.replace(/```[\s\S]*?```/g, '').replace(/[#*>\-|]/g, '');
	const cnChars = (cleaned.match(/[\u4e00-\u9fff\u3400-\u4dbf]/g) || []).length;
	const enWords = cleaned
		.replace(/[\u4e00-\u9fff\u3400-\u4dbf]/g, '')
		.split(/\s+/)
		.filter(Boolean).length;

	const cnMin = cnChars / WORDS_PER_MINUTE_CN;
	const enMin = enWords / WORDS_PER_MINUTE_EN;
	const total = Math.max(1, Math.ceil(cnMin + enMin));
	return total;
}
