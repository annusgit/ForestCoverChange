
/*
  This file will be used to download our sentinel-2 dataset
*/
// start by creating an AOI first, we'll do it using a polygon instance from geometry
var pakistan = /* color: #98ff00 */ee.Geometry.Polygon(
        [[[73.65007590061041, 33.924912470572],
          [74.31637228732916, 33.924912470572],
          [74.31637228732916, 34.44520782244974],
          [73.65007590061041, 34.44520782244974]]]);

var germany = /* color: #d63000 */ee.Geometry.Polygon(
        [[[7.7552313346861865, 49.278817914014034],
          [8.105420543670562, 49.27971381160097],
          [8.104047252654937, 49.46659938059896],
          [7.7634710807799365, 49.46570688270015]]]);

// Draw it on the map to see what we get on the map
// Map.addLayer(germany);

// some utilities

// crop function for croping out our main region of interest
function cropout_aoi(image){
  return image.clip(this_aoi); // but this is the one that actually does the cropping
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

function scale(image, div) {
  return image.float().divide(div).multiply(255).uint8(); //.cast(ee.uint8);
}


// declare vars here
var all_bands = ['B1', 'B2', 'B3',
                  'B4', 'B5', 'B6',
                  'B7', 'B8', 'B8A',
                  'B9', 'B10', 'B11',
                  'B12'];
var this_aoi = pakistan;
var landsat_7 = 'LANDSAT/LC08/C01/T1';
// var landsat_7 = 'LANDSAT/LE07/C01/T1_RT';
// var landsat_7 = 'LANDSAT/LE07/C01/T1_RT';
var show_bands = ['B3', 'B2', 'B1'];
var export_bands = ('B4', 'B3', 'B2');
var this_max = 20000;


// the following function returns mosaiced images, that are actually complete
function get_selected_dataset(){
  var count  = 1;
  var month; var month_interval = 11; var month_end = 11; // download one image for 6-month interval
  var year; var year_interval = 1; var year_end = 2019;
  for (year = 2016; year < year_end; year+=year_interval) {
    for (month = 01; month < month_end; month+=month_interval) {
      var next_month = month + month_interval;
      print('currently on (' + month + '/' + year + ')->(' + next_month + '/' + year + ')');
      // Now define the dates we want to get data on
      var date_from = ee.Date({date: year + '-' + month + '-01'});
      var date_to = ee.Date({date: year + '-' + next_month + '-01'});
      // import sentinel images, clip your area of interest, dates and cloud cover as needed
      var image_collection = ee.ImageCollection(landsat_7) //
                            // .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.5))
                            .filterDate({start:date_from, end:date_to})
                            .filterBounds(this_aoi) // locates ones close to this polygon
                            .map(cropout_aoi); // actually does the clipping part
                            // .map(maskS2clouds)
      var total_count = image_collection.size();
      print('log: total images in collection =', total_count);
      var this_image = image_collection.median();
      Map.addLayer(this_image, {bands: show_bands, max: this_max});
      Export.image.toDrive({
                            image: this_image.select(export_bands),
                            description: 'German_S2_rgb_' + count,
                            folder: 'germany_s2',
                            fileNamePrefix: 'g_rgb_' + count,
                            // dimensions,
                            region: this_aoi,
                            scale: 10
                            });
      count += 1;
    }
  }
}


// the following function returns all images in a bad form, because all of them are incomplete
function get_all_images(){
  var date_from = ee.Date({date: '2018-06-01'});
  var date_to = ee.Date({date: '2018-07-31'});

  // import sentinel images, clip your area of interest, dates and cloud cover as needed
  var image_collection = ee.ImageCollection('COPERNICUS/S2')
                          // .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                          .filterDate({start:date_from, end:date_to})
                          .filterBounds(rawalLake) // locates ones close to this polygon
                          .map(cropout_aoi) // actually does the clipping part
                          .map(maskS2clouds);
  var all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];
  var total_count = image_collection.size();
  var count = 1;
  print('log: total images in collection =', total_count);
  function exporter(image)
  {
    try{
    Export.image.toDrive({
                        image: image.select('B2', 'B3', 'B4'),
                        description: 'rawalLake' + count,
                        folder: 'rawalLake',
                        fileNamePrefix: 'rawalLake_' + count,
                        // dimensions,
                        region: rawalLake,
                        scale: 10
                        });
    print('exported ' + count);
    count += 1;
    // Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0, max: 5000});
    }
    catch(err) {
        print('INFO: something bad happened here');
    }
    finally {
    return 1;
    }
  }
  // image_collection.map(exporter);
  // Map.addLayer(rawalLake);
  Map.addLayer(image_collection);
  var total_found = 6;
  var colList = image_collection.toList(total_found);
  for (var i=0; i < total_found; i++){
    var img = ee.Image(colList.get(i));
    print(img)
    exporter(img);
    Map.addLayer(img, {bands: ['B4', 'B3', 'B2'], min: 0, max: 5000});
  }
}


/*////////////////////////////////////////////////////////////////////////////////////////
//  make function calls beyond this point                                                //
*/////////////////////////////////////////////////////////////////////////////////////////
get_selected_dataset();


// random codes for rawal lake
// var date_from = ee.Date({date: '2018-08-01'});
// var date_to = ee.Date({date: '2018-08-31'});
// // import sentinel images, clip your area of interest, dates and cloud cover as needed
// var image_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT') //
//                       // .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
//                       .filterDate({start:date_from, end:date_to})
//                       .filterBounds(rawalLake) // locates ones close to this polygon
//                       .map(cropout_aoi) // actually does the clipping part
//                       // .map(maskS2clouds);
// var all_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'];
// var total_count = image_collection.size();
// print('log: total images in collection =', total_count);
// var this_image = image_collection.median();
// // this_image = scale(this_image, 10);
// Map.addLayer(this_image, {bands: ['B3', 'B2', 'B1']});
// Export.image.toDrive({
//                       image: this_image.select('B3', 'B2', 'B1'),
//                       description: 'landsat_rawal',
//                       folder: 'landsat_rawal',
//                       fileNamePrefix: 'landsat_rawal_august',
//                       // dimensions,
//                       region: rawalLake,
//                       scale: 10
//                       });






