param([string]$InputDoc,[string]$OutputPdf)
$word = New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
try {
 $doc=$word.Documents.Open($InputDoc)
 $doc.Fields.Update() | Out-Null
 foreach($toc in $doc.TablesOfContents) { $toc.Update(); $toc.Range.Font.Name="Times New Roman"; $toc.Range.Font.Size=14; $toc.Range.ParagraphFormat.LineSpacingRule=1; $toc.Range.ParagraphFormat.SpaceBefore=0; $toc.Range.ParagraphFormat.SpaceAfter=0 }
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
 $doc.Save()
 $doc.ExportAsFixedFormat($OutputPdf,17)
 Write-Output ('Pages: '+$doc.ComputeStatistics(2))
 $doc.Close(0)
} finally { $word.Quit() }

