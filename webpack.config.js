const path = require('path');

module.exports = {
  mode: process.env.NODE_ENV === 'production' ? 'production' : 'development',
  entry: './ml-book/assets/js/main.js',
  output: {
    filename: 'app.js',
    path: path.resolve(__dirname, 'ml-book', 'assets', 'js'),
  },
};
