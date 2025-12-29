1. SIFT
   ```
   sift = cv.SIFT_create()

   kp = sift.detect(gray,None)
   kp,des = sift.compute(gray,kp)  #kp, des = sift.detectAndCompute(gray, None)

   img=cv.drawKeypoints(gray,kp,img)
   print(f'Number of keypoints detected: {len(kp)}')
   ```
2. BRIEF
   ```
    # 初始化 FAST 检测器
    star = cv.xfeatures2d.StarDetector_create()
    # 初始化 BRIEF 提取器
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    # 使用 STAR 查找关键点
    kp = star.detect(img,None)
    # 使用 BRIEF 计算描述符
    kp, des = brief.compute(img, kp)
   ```
3. ORB
   ```
    # 初始化 ORB 检测器
    orb = cv.ORB_create()
    # 使用ORB找到关键点
    kp = orb.detect(img,None)
    # 使用ORB计算描述符
    kp, des = orb.compute(img, kp)
   ```