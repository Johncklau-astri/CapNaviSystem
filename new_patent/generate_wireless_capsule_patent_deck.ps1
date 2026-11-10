$ErrorActionPreference = 'Stop'

function Add-Bullets {
    param(
        $Slide,
        [string[]]$Lines,
        [float]$Left,
        [float]$Top,
        [float]$Width,
        [float]$Height,
        [int]$FontSize = 22
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $textRange = $shape.TextFrame.TextRange
    $textRange.Text = ($Lines -join "`r`n")
    $textRange.Font.Name = 'Aptos'
    $textRange.Font.Size = $FontSize
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0

    for ($index = 1; $index -le $textRange.Paragraphs().Count; $index++) {
        $paragraph = $textRange.Paragraphs($index)
        $paragraph.ParagraphFormat.Bullet.Visible = -1
        $paragraph.ParagraphFormat.Bullet.Character = 8226
        $paragraph.ParagraphFormat.SpaceAfter = 6
    }

    return $shape
}

function Add-Title {
    param(
        $Slide,
        [string]$Text
    )

    $shape = $Slide.Shapes.AddTextbox(1, 36, 18, 620, 48)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = 'Aptos Display'
    $shape.TextFrame.TextRange.Font.Size = 28
    $shape.TextFrame.TextRange.Font.Bold = -1
    return $shape
}

function Add-Subtitle {
    param(
        $Slide,
        [string[]]$Lines
    )

    $shape = $Slide.Shapes.AddTextbox(1, 56, 160, 560, 180)
    $shape.TextFrame.TextRange.Text = ($Lines -join "`r`n")
    $shape.TextFrame.TextRange.Font.Name = 'Aptos'
    $shape.TextFrame.TextRange.Font.Size = 20
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    return $shape
}

function Add-SectionTag {
    param(
        $Slide,
        [string]$Text
    )

    $tag = $Slide.Shapes.AddShape(1, 540, 20, 160, 26)
    $tag.Fill.ForeColor.RGB = 13421772
    $tag.Line.Visible = 0
    $tag.TextFrame.TextRange.Text = $Text
    $tag.TextFrame.TextRange.Font.Name = 'Aptos'
    $tag.TextFrame.TextRange.Font.Size = 11
    $tag.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    return $tag
}

function Add-Box {
    param(
        $Slide,
        [float]$Left,
        [float]$Top,
        [float]$Width,
        [float]$Height,
        [string]$Text,
        [int]$FillColor,
        [int]$FontSize = 16
    )

    $shape = $Slide.Shapes.AddShape(1, $Left, $Top, $Width, $Height)
    $shape.Fill.ForeColor.RGB = $FillColor
    $shape.Line.ForeColor.RGB = 8947848
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = 'Aptos'
    $shape.TextFrame.TextRange.Font.Size = $FontSize
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    return $shape
}

function Add-Arrow {
    param(
        $Slide,
        [float]$Left,
        [float]$Top,
        [float]$Width,
        [float]$Height
    )

    $shape = $Slide.Shapes.AddShape(33, $Left, $Top, $Width, $Height)
    $shape.Fill.ForeColor.RGB = 10066329
    $shape.Line.ForeColor.RGB = 8947848
    return $shape
}

function Clear-SlideShapes {
    param($Slide)

    for ($index = $Slide.Shapes.Count; $index -ge 1; $index--) {
        $Slide.Shapes.Item($index).Delete()
    }
}

$templatePath = Join-Path $PSScriptRoot 'CapAutoSystemPatent.ppt'
$outputPath = Join-Path $PSScriptRoot 'Wireless_Magnetic_Capsule_Endoscopy_Patent_Deck.pptx'

if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Template not found: $templatePath"
}

$slides = @(
    @{ Title = 'Wireless Magnetic Capsule Endoscopy System'; Type = 'title'; Lines = @('September 2026', 'Draft patent concept for stomach examination', 'Wireless power + low-cost magnetic actuation') },
    @{ Title = 'Background'; Tag = 'Overview'; Bullets = @('Key embodiments of the invention', 'Wireless power and magnetic actuation architecture', 'Major claims and differences with prior art', 'Business value and filing direction') },
    @{ Title = 'Background'; Tag = 'Clinical Need'; Bullets = @('Conventional stomach capsule systems need bulky external robots or internal batteries with limited runtime', 'Dedicated actuation hardware and separate power subsystems increase footprint, transport difficulty, and cost', 'A simpler system can target two gross stomach viewpoints and recover coverage by steering the capsule viewing axis locally') },
    @{ Title = 'Key Embodiments of the Invention (1/4)'; Tag = 'System'; Bullets = @('Capsule contains three orthogonal coil pairs, camera, LEDs, control ASIC or MCU, inertial sensing, and a small energy buffer', 'The same tri-axial coil set is used for two functions: receiving wireless power and generating a controllable magnetic dipole', 'Two external body-side coils create a one-dimensional left/right gradient to shuttle the capsule between opposing stomach regions') },
    @{ Title = 'Key Embodiments of the Invention (2/4)'; Tag = 'Capsule'; Bullets = @('Each coil axis is tuned with a resonant capacitor and connected to an active rectifier so the best-coupled axis can harvest power regardless of orientation', 'Harvested energy charges a supercapacitor or rechargeable micro-cell that supports LED pulses, imaging bursts, and current drive to the coils', 'The on-board controller energizes the internal coil axes with commanded current ratios to synthesize an arbitrary magnetic moment vector for lens steering') },
    @{ Title = 'Key Embodiments of the Invention (3/4)'; Tag = 'External'; Bullets = @('The outside hardware is reduced to a bilateral one-dimensional coil fixture placed at the patient left and right sides', 'The fixture produces two dominant gradient states: pull-left and pull-right, enabling repeatable shuttling between two gross stomach inspection zones', 'Mechanical simplicity reduces cost, installation complexity, and the need for multi-axis robotic positioning') },
    @{ Title = 'Key Embodiments of the Invention (4/4)'; Tag = 'Workflow'; Bullets = @('At each gross location the capsule remains approximately in place while internal tri-axial actuation changes the camera viewing direction', 'Sector-by-sector viewing captures fundus, body, antrum, and curvature regions without rotating the patient or moving a large external magnet arm', 'The concept is best positioned as a low-cost stomach screening platform rather than a full free-space 6-DOF navigation system') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (1/6)'; Tag = 'Claim 1'; Bullets = @('A swallowable endoscopy capsule with three orthogonal coil pairs configured both as wireless power receivers and as magnetic actuation coils', 'Prior art usually separates the battery or receive coil from the magnetic element; here the tri-axial coil network is dual-use', 'This reduces internal component count and supports orientation-independent energy capture') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (2/6)'; Tag = 'Claim 2'; Bullets = @('A bilateral one-dimensional external coil system that generates only two principal gradient states for gross left/right shuttling inside the stomach', 'The external hardware is substantially simpler than robotic permanent-magnet manipulators or multi-axis Helmholtz/Maxwell arrays', 'Inspection completeness is recovered by local optical steering rather than full external pose control') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (3/6)'; Tag = 'Claim 3'; Bullets = @('A control method in which the capsule synthesizes an arbitrary internal magnetic moment vector using the three orthogonal coil pairs', 'The internal moment interacts with the external bias or gradient field to tilt the optical axis toward selected stomach sections', 'This separates coarse capsule translation from fine view steering') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (4/6)'; Tag = 'Claim 4'; Bullets = @('A multiplexed operating mode that decouples low-frequency actuation and high-frequency wireless power transfer', 'Recommended implementation: quasi-static or sub-20 Hz gradient commands for shuttling, plus resonant inductive power transfer near 150 to 300 kHz', 'Filtering and time-division bursts allow the shared coil hardware to support both functions without severe cross-coupling') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (5/6)'; Tag = 'Claim 5'; Bullets = @('Wireless power should target roughly 200 to 500 mW average delivered power for camera, illumination, sensing, and moment generation', 'A practical starting point is around 200 kHz because the near-field penetrates tissue well, capsule-size resonant coils remain feasible, and power electronics stay moderate in complexity', 'Power bursts can charge a local buffer capacitor so imaging and steering remain stable even when coupling changes with orientation') },
    @{ Title = 'Major Claims and Differences with the Prior Arts (6/6)'; Tag = 'Claim 6'; Bullets = @('Closed-loop localization can use capsule IMU data, image features, magnetic sensing, or external field estimation to confirm which stomach sectors have already been observed', 'Safety controls should bound tissue heating, induced current density, and internal coil temperature while enforcing maximum capsule dipole current', 'The strongest patent position is the low-hardware architecture plus the shared-coil operating method, not just the raw hardware list') },
    @{ Title = 'Major claims'; Tag = 'Assessment'; Bullets = @('Claim family A: capsule structure with dual-use tri-axial coil pairs, resonant receive path, and vector magnetic actuation path', 'Claim family B: bilateral one-dimensional external coil fixture with discrete gradient states for shuttle-style stomach inspection', 'Claim family C: operating method for switching between power-transfer mode and view-steering mode using a shared coil set', 'Claim family D: inspection workflow based on two or more gross docking positions plus local optical-axis steering for coverage completion') },
    @{ Title = 'Inventorship detail per claim'; Tag = 'To Allocate'; Bullets = @('Inventorship should follow who conceived: the shared-coil capsule structure, the bilateral one-dimensional field source, the multiplexed power/actuation control method, and the two-point scan workflow', 'If multiple contributors refined only implementation details after the core concept was defined, they may not map to every claim set', 'A claim chart should be completed with dated conception records before filing') },
    @{ Title = 'Patent Portfolio'; Tag = 'Portfolio Fit'; Bullets = @('This concept extends a magnetic capsule portfolio from bulky external actuation toward a portable and lower-cost configuration', 'It complements prior inspection-system patents by claiming the shared-coil capsule and simplified external field source', 'Follow-on filings could separately cover localization, wireless power electronics, and coverage-planning algorithms') },
    @{ Title = 'Business Value'; Tag = 'Value'; Bullets = @('Minimal outside hardware lowers manufacturing cost, room footprint, and service complexity', 'Wireless power can remove or shrink the internal battery, improving runtime and enabling brighter illumination or active magnetic steering', 'A lower-cost screening platform could fit outpatient, mobile, or regional-clinic deployments where large robotic systems are impractical') },
    @{ Title = 'Which Jurisdiction to File'; Tag = 'Filing'; Bullets = @('File first where claim scope and medical-device investment value are highest: typically US as the anchor jurisdiction', 'Consider parallel CN filing if manufacturing, market access, or local enforcement matters; EP can follow if broad hospital deployment is expected', 'Before filing, run a prior-art search focused on magnetically actuated capsule endoscopy, inductive power transfer capsules, and tri-axial receive coil systems') },
    @{ Title = 'Thank You'; Type = 'closing'; Lines = @('Assessment: technically plausible with careful multiplexing and thermal control', 'Recommended next step: prior-art search plus first-order magnetic and power budget simulation', 'Draft generated from existing patent template') }
)

$powerPoint = $null
$presentation = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Open($templatePath, $true, $false, $false)

    if ($presentation.Slides.Count -lt $slides.Count) {
        throw "Template has $($presentation.Slides.Count) slides; expected at least $($slides.Count)."
    }

    for ($index = 1; $index -le $slides.Count; $index++) {
        $slide = $presentation.Slides.Item($index)
        Clear-SlideShapes -Slide $slide

        $spec = $slides[$index - 1]
        Add-Title -Slide $slide -Text $spec.Title | Out-Null

        if ($spec.ContainsKey('Tag')) {
            Add-SectionTag -Slide $slide -Text $spec.Tag | Out-Null
        }

        if ($spec.Type -eq 'title') {
            Add-Subtitle -Slide $slide -Lines $spec.Lines | Out-Null
            Add-Box -Slide $slide -Left 140 -Top 320 -Width 440 -Height 46 -Text 'Portable low-cost stomach inspection concept' -FillColor 15132390 -FontSize 18 | Out-Null
            continue
        }

        if ($spec.Type -eq 'closing') {
            Add-Subtitle -Slide $slide -Lines $spec.Lines | Out-Null
            continue
        }

        Add-Bullets -Slide $slide -Lines $spec.Bullets -Left 54 -Top 92 -Width 600 -Height 270 -FontSize 22 | Out-Null

        switch ($index) {
            4 {
                Add-Box -Slide $slide -Left 70 -Top 315 -Width 150 -Height 64 -Text 'External left coil' -FillColor 15658734 | Out-Null
                Add-Arrow -Slide $slide -Left 225 -Top 332 -Width 70 -Height 28 | Out-Null
                Add-Box -Slide $slide -Left 300 -Top 300 -Width 140 -Height 92 -Text 'Capsule\n3 orthogonal\ncoil pairs' -FillColor 14548957 | Out-Null
                Add-Arrow -Slide $slide -Left 445 -Top 332 -Width 70 -Height 28 | Out-Null
                Add-Box -Slide $slide -Left 520 -Top 315 -Width 150 -Height 64 -Text 'External right coil' -FillColor 15658734 | Out-Null
            }
            5 {
                Add-Box -Slide $slide -Left 55 -Top 308 -Width 180 -Height 72 -Text 'Tri-axial resonant coils' -FillColor 14548957 | Out-Null
                Add-Arrow -Slide $slide -Left 238 -Top 330 -Width 56 -Height 24 | Out-Null
                Add-Box -Slide $slide -Left 292 -Top 308 -Width 140 -Height 72 -Text 'Active rectifier\nand filter' -FillColor 15132390 | Out-Null
                Add-Arrow -Slide $slide -Left 435 -Top 330 -Width 56 -Height 24 | Out-Null
                Add-Box -Slide $slide -Left 488 -Top 308 -Width 176 -Height 72 -Text 'Supercap + MCU +\ncoil current drivers' -FillColor 15658734 | Out-Null
            }
            6 {
                Add-Box -Slide $slide -Left 92 -Top 320 -Width 160 -Height 60 -Text 'Gradient state A\nPull-left' -FillColor 15658734 | Out-Null
                Add-Box -Slide $slide -Left 450 -Top 320 -Width 160 -Height 60 -Text 'Gradient state B\nPull-right' -FillColor 15658734 | Out-Null
                Add-Arrow -Slide $slide -Left 258 -Top 334 -Width 80 -Height 28 | Out-Null
                Add-Box -Slide $slide -Left 340 -Top 314 -Width 105 -Height 72 -Text 'Stomach\nworkspace' -FillColor 14548957 | Out-Null
            }
            11 {
                Add-Box -Slide $slide -Left 72 -Top 302 -Width 170 -Height 74 -Text 'Actuation band\n0 to 20 Hz' -FillColor 15658734 | Out-Null
                Add-Arrow -Slide $slide -Left 245 -Top 326 -Width 80 -Height 26 | Out-Null
                Add-Box -Slide $slide -Left 330 -Top 302 -Width 170 -Height 74 -Text 'Power band\n150 to 300 kHz' -FillColor 14548957 | Out-Null
                Add-Arrow -Slide $slide -Left 505 -Top 326 -Width 80 -Height 26 | Out-Null
                Add-Box -Slide $slide -Left 590 -Top 302 -Width 90 -Height 74 -Text 'Shared\ncoils' -FillColor 15132390 | Out-Null
            }
            12 {
                Add-Box -Slide $slide -Left 80 -Top 302 -Width 160 -Height 66 -Text 'Camera + MCU\n40 to 120 mW' -FillColor 15132390 | Out-Null
                Add-Box -Slide $slide -Left 270 -Top 302 -Width 160 -Height 66 -Text 'LED average\n50 to 200 mW' -FillColor 14548957 | Out-Null
                Add-Box -Slide $slide -Left 460 -Top 302 -Width 170 -Height 66 -Text 'Coil drive + sensing\n50 to 180 mW' -FillColor 15658734 | Out-Null
            }
            17 {
                Add-Box -Slide $slide -Left 70 -Top 316 -Width 180 -Height 64 -Text 'Low capital cost' -FillColor 14548957 | Out-Null
                Add-Box -Slide $slide -Left 270 -Top 316 -Width 180 -Height 64 -Text 'Portable deployment' -FillColor 15132390 | Out-Null
                Add-Box -Slide $slide -Left 470 -Top 316 -Width 180 -Height 64 -Text 'Longer usable runtime' -FillColor 15658734 | Out-Null
            }
        }
    }

    $presentation.SaveAs($outputPath, 24)
    $presentation.Close()
    $powerPoint.Quit()

    Write-Output "CREATED: $outputPath"
}
finally {
    if ($presentation) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($powerPoint) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
}