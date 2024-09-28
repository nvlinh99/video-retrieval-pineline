// Function to download the CSV file
const downloadCSV = (data: string, fileName?: string): void => {
  const blob = new Blob([data], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.setAttribute("href", url);
  a.setAttribute("download", fileName || "query.csv");
  a.click();
};

function csvmaker<T>(items: T[], getRow: (item: T) => string) {
  const csvRows: string[] = [];

  for (const item of items) {
    const row = getRow(item);

    if (row) {
      csvRows.push(row);
    }
  }

  return csvRows.join("\n");
}

// Main function to generate the CSV and trigger the download
function exportItemsCSV<T>(items: T[], getRow: (item: T) => string) {
  const csvData = csvmaker(items, getRow);
  downloadCSV(csvData);
}

export default exportItemsCSV;
