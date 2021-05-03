var marked = require('marked');
var fs = require('fs');

var knnMd = fs.readFileSync('K_Nearest_Neighbors.md', 'utf-8');
var knn = marked(knnMd);

fs.writeFileSync('knn.html', knn);