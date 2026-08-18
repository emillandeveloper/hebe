function focusMainWindow(mainWindow) {
  if (!mainWindow || mainWindow.isDestroyed?.()) return false;
  if (mainWindow.isMinimized?.()) mainWindow.restore();
  if (!mainWindow.isVisible?.()) mainWindow.show();
  mainWindow.focus();
  return true;
}

module.exports = { focusMainWindow };
