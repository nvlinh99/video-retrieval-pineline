export function removeFileExtension(filename = "") {
  return filename.substring(0, filename.lastIndexOf(".")) || filename;
}
