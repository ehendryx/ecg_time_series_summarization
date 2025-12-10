function class_count = condense_soybean(ls)

  class_count = zeros(1,16);

  len = length(ls);

  for i = 1:len
    switch(ls{i})
      case "alternarialeaf-spot"
        class_count(1) = class_count(1) + 1;
      case "anthracnose"
        class_count(2) = class_count(2) + 1;
      case "bacterial-blight"
        class_count(3) = class_count(3) + 1;
      case "bacterial-pustule"
        class_count(4) = class_count(4) + 1;
      case "brown-spot"
        class_count(5) = class_count(5) + 1;
      case "brown-stem-rot"
        class_count(6) = class_count(6) + 1;
      case "charcoal-rot"
        class_count(7) = class_count(7) + 1;
      case "diaporthe-stem-canker"
        class_count(8) = class_count(8) + 1;
      case "downy-mildew"
        class_count(9) = class_count(9) + 1;
      case "frog-eye-leaf-spot"
        class_count(10) = class_count(10) + 1;
      case "phyllosticta-leaf-spot"
        class_count(11) = class_count(11) + 1;
      case "phytophthora-rot"
        class_count(12) = class_count(12) + 1;
      case "powdery-mildew"
        class_count(13) = class_count(13) + 1;
      case "purple-seed-stain"
        class_count(14) = class_count(14) + 1;
      case "rhizoctonia-root-rot"
        class_count(15) = class_count(15) + 1;
      otherwise
        class_count(16) = class_count(16) + 1;
  end
end

