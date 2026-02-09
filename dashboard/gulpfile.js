var gulp = require('gulp');
var path = require('path');
var sass = require('gulp-sass')(require('sass'));
var autoprefixer = require('gulp-autoprefixer');
var sourcemaps = require('gulp-sourcemaps');
var open = require('gulp-open');
var clean = require('gulp-clean');
var { createProxyMiddleware } = require('http-proxy-middleware');
var browserSync = require('browser-sync').create();

var Paths = {
  HERE: './',
  DIST: 'dist/',
  CSS: './assets/css/',
  SCSS_TOOLKIT_SOURCES: './assets/scss/paper-dashboard.scss',
  SCSS: './assets/scss/**/**'
};

gulp.task('compile-scss', function () {
  return gulp.src(Paths.SCSS_TOOLKIT_SOURCES)
    .pipe(sourcemaps.init())
    .pipe(sass().on('error', sass.logError))
    .pipe(autoprefixer())
    .pipe(sourcemaps.write(Paths.HERE))
    .pipe(gulp.dest(Paths.CSS))
    .pipe(browserSync.stream());
});

gulp.task('watch', function () {
  gulp.watch(Paths.SCSS, gulp.series('compile-scss'));
  gulp.watch('examples/*.html').on('change', browserSync.reload);
  gulp.watch('*.html').on('change', browserSync.reload);
});

gulp.task('serve', function () {
  browserSync.init({
    server: {
      baseDir: "./",
      middleware: [
        createProxyMiddleware({
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          pathFilter: '/api',
          pathRewrite: {
            '^/api': '' // remove /api prefix when forwarding
          }
        })
      ]
    }
  });

  gulp.watch(Paths.SCSS, gulp.series('compile-scss'));
  gulp.watch('examples/*.html').on('change', browserSync.reload);
  gulp.watch('*.html').on('change', browserSync.reload);
});

gulp.task('open', function () {
  gulp.src('examples/dashboard.html')
    .pipe(open());
});

gulp.task('open-app', gulp.parallel('open', 'watch'));

gulp.task('clean', function () {
  return gulp.src('dist', { read: false, allowEmpty: true })
    .pipe(clean());
});

gulp.task('build', gulp.series('clean', 'compile-scss', function () {
  return gulp.src([
    '*.html',
    'assets/**/*',
    'examples/**/*',
    'docs/**/*'
  ], { base: './' })
    .pipe(gulp.dest('dist'));
}));