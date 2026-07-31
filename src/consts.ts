// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

export const SITE_TITLE = "Yukii's Lab";
export const SITE_DESCRIPTION = '记录技术、思考与生活';
export const AUTHOR_NAME = 'Yukii';
export const AUTHOR_BIO = '开发者 / 写点东西 / 偶尔折腾';

export const CATEGORIES = [
	{ slug: 'notes', label: '技术笔记' },
	{ slug: 'fixes', label: '疑难处置' },
	{ slug: 'essays', label: '随笔' },
] as const;
export type CategorySlug = (typeof CATEGORIES)[number]['slug'];
export const categoryLabel = (slug: string) =>
	CATEGORIES.find((c) => c.slug === slug)?.label ?? slug;
