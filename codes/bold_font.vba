Sub bold_font()
    Dim ws As Worksheet
    Dim rng As Range
    Dim row As Range
    Dim cell As Range
    Dim minVal As Double
    Dim firstMin As Range
    
    ' Set the worksheet (change "cmpResult" to your actual sheet name if different)
    Set ws = ThisWorkbook.Worksheets("cmpResult")
    
    ' Set the range from C2 to F61
    Set rng = ws.Range("C2:F61")
    
    ' Loop through each row in the range
    For Each row In rng.Rows
        ' Reset variables for each row
        minVal = 999999 ' Initialize with a very large number
        Set firstMin = Nothing
        
        ' Find the minimum value in the row
        For Each cell In row.Cells
            If IsNumeric(cell.Value) Then
                If cell.Value < minVal Then
                    minVal = cell.Value
                    Set firstMin = cell
                End If
            End If
        Next cell
        
        ' Bold all cells with the minimum value (in case of ties)
        If Not firstMin Is Nothing Then
            For Each cell In row.Cells
                If IsNumeric(cell.Value) And cell.Value = minVal Then
                    cell.Font.Bold = True
                Else
                    cell.Font.Bold = False ' Optional: unbold others
                End If
            Next cell
        End If
    Next row
    
    MsgBox "Minimum values in each row have been bolded.", vbInformation
End Sub
