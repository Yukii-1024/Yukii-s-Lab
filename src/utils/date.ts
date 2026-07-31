export function formatDate(date: Date): string {
	const y = date.getFullYear();
	const m = date.getMonth() + 1;
	const d = date.getDate();
	return `${y}/${m}/${d}`;
}
