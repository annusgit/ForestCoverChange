

// you should paste this code into your google earth engine code editor to download images from 2016-2018
/*
  This file will be used to download our sentinel-2 dataset
*/
// start by creating an AOI first, we'll do it using a polygon instance from geometry
var pakistan = /* color: #98ff00 */ee.Geometry.Polygon(
        [[[73.65007590061041, 33.924912470572],
          [74.31637228732916, 33.924912470572],
          [74.31637228732916, 34.44520782244974],
          [73.65007590061041, 34.44520782244974]]]);

var germany = /* color: #0b4a8b */ee.Geometry.Polygon(
        [[[7.656634256525308, 49.45964681885849],
          [8.045275613947183, 49.45964681885849],
          [8.043902322931558, 49.67251796515247],
          [7.659380838556558, 49.67251796515247]]]);

// Draw it on the map to see what we get on the map
Map.addLayer(germany);

// some utilities

// crop function for croping out our main region of interest
function cropout_aoi(image){
  return image.clip(germany); // but this is the one that actually does the cropping
}
/*
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
*/
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask);//.divide(10000);
}

// scale for our needs
function scale(image) {
  return image.divide(4096).multiply(255).uint8(); //.cast(ee.uint8);
}

// create python like string format function (stackoverflow)
// String.prototype.format = function () {
//   var i = 0, args = arguments;
//   return this.replace(/{}/g, function () {
//     return typeof args[i] != 'undefined' ? args[i++] : '';
//   });
// };

var i = 0;
var total_count = 1;
var month = 01; var month_interval = 5; var month_end = 11; // download one image for 6-month interval
var year; var year_end = 2019;
for (year = 2016; year < year_end; year++) {
  for (month = 01; month < month_end; month+=month_interval+1) {
    // if ((month === 0) && (year === 2018))
        // break;
    var next_month = month + month_interval;
    print('currently on (' + month + '/' + year + ')->(' + next_month + '/' + year + ') => total count = ' + total_count);
    // Now define the dates we want to get data on
    var date_from = ee.Date({date: year + '-' + month + '-01'});
    var date_to = ee.Date({date: year + '-' + next_month + '-01'});

    // import sentinel images for your area of interest
    // and dates of interest and cloud cover as needed
    var image_collection = ee.ImageCollection('COPERNICUS/S2')
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                            .filterDate({start:date_from, end:date_to})
                            .filterBounds(germany) // locates ones close to this polygon
                            .map(cropout_aoi)
                            // .map(function(image){return image.clip(germany)}) // done above
                            .map(maskS2clouds);
                            // .map(rescale);
                            // .cast({'B4':ee.uint8, 'B3':ee.uint8, 'B2':ee.uint8});


    // see how many images we got
    var count = image_collection.size();
    // print(typeof(count))
    if (count == 0)
    {
      print('log: missing this interval');
      continue;
    }
    print('log: total images in collection =', count);

    var image_median = image_collection.median();

    var all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];

    // Export.image.toDrive({
    //                       image: image_median.select(all_bands),
    //                       description: 'None',
    //                       region:germany
    //                     })
    Export.image.toDrive({
                          image: image_median.select(all_bands),
                          description: 'German_S2_' + total_count,
                          folder: 'germany_s2',
                          fileNamePrefix: 'g_' + total_count,
                          // dimensions,
                          region: germany,
                          scale: 10
                          });
    total_count += 1;
  }
}









